from flask import Flask, jsonify, render_template
from threading import Thread, Lock
import time
import pandas as pd
import os
from pathlib import Path

from prediction_manager import run_prediction
from engines.image_analysis_simulation import ImageAnalysisSimulation
from engines.wavefront_reconstruction_engine import WavefrontReconstructionEngine
from engines.turbulence_estimation_engine import TurbulenceEstimationEngine
from engines.dm_translation_engine import DMTranslationEngine

############################################################
# Flask App
############################################################

app = Flask(__name__)

############################################################
# Initialize Engines
############################################################

image_engine = ImageAnalysisSimulation(grid_size=8, noise_level=0.02)
wavefront_engine = WavefrontReconstructionEngine(grid_size=8)
turbulence_engine = TurbulenceEstimationEngine()
dm_engine = DMTranslationEngine(actuator_grid=8)

############################################################
# Shared Data + Lock
############################################################

data_lock = Lock()

latest_data = {
    "frame": 0,
    "shift_data": [],
    "wavefront": [],
    "zernike": {},
    "turbulence": {},
    "dm_commands": {},
    "prediction": None
}

# Separate store for the latest prediction result (updated asynchronously)
latest_prediction = {"result": None}
prediction_lock = Lock()
prediction_running = False  # Guard to prevent overlapping prediction runs

############################################################
# Async Prediction Worker
############################################################

def prediction_worker():
    """Runs in its own thread. Loops independently, updating prediction results."""
    global prediction_running
    while True:
        prediction_running = True
        try:
            result = run_prediction()
        except Exception as e:
            error_msg = str(e).lower()
            if "column" in error_msg or "no columns" in error_msg:
                result = {"status": "Training model, please wait..."}
            else:
                print(f"Error running prediction: {e}")
                result = {"error": str(e)}
        finally:
            prediction_running = False

        with prediction_lock:
            latest_prediction["result"] = result

        # Tune this interval to however often you want predictions refreshed.
        # The AO pipeline is completely unaffected by this sleep.
        time.sleep(1)

############################################################
# Continuous AO Pipeline
############################################################

def adaptive_optics_pipeline():
    global latest_data
    frame = 0
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    csv_path = PROJECT_ROOT / "data" / "synthetic" / "zernike_timeseries.csv"

    while True:
        ####################################################
        # Image Analysis Simulation
        ####################################################
        shift_data = image_engine.generate_shift_frame()

        ####################################################
        # Wavefront Reconstruction
        ####################################################
        wavefront, zernike = wavefront_engine.process(shift_data)

        ####################################################
        # Update CSV (Rolling Window)
        ####################################################
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                new_row = {
                    "frame": frame,
                    "z1": zernike.get("a1", 0.0),
                    "z2": zernike.get("a2", 0.0),
                    "z3": zernike.get("a3", 0.0),
                    "z4": zernike.get("a4", 0.0),
                    "z5": zernike.get("a5", 0.0),
                    "z6": zernike.get("a6", 0.0)
                }
                df = pd.concat([df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error updating CSV: {e}")

        ####################################################
        # Turbulence Estimation
        ####################################################
        turbulence_engine.add_frame(zernike)
        turbulence = {}
        if len(turbulence_engine.history) >= 2:
            turbulence = turbulence_engine.process()

        ####################################################
        # DM Translation
        ####################################################
        dm_commands = dm_engine.process(wavefront, zernike)

        ####################################################
        # Grab latest prediction result (non-blocking)
        ####################################################
        with prediction_lock:
            current_prediction = latest_prediction["result"]

        ####################################################
        # Store Latest Results
        ####################################################
        with data_lock:
            latest_data = {
                "frame": frame,
                "shift_data": shift_data,
                "wavefront": wavefront.tolist(),
                "zernike": zernike,
                "turbulence": turbulence,
                "dm_commands": dm_commands,
                "prediction": current_prediction
            }

        frame += 1
        time.sleep(1)

############################################################
# Flask Routes
############################################################

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/data")
def data():
    with data_lock:
        return jsonify(latest_data)

############################################################
# Start Background Threads
############################################################

Thread(target=adaptive_optics_pipeline, daemon=True).start()
Thread(target=prediction_worker, daemon=True).start()

############################################################

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)