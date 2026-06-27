from flask import Flask, jsonify, render_template
from threading import Thread
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

image_engine = ImageAnalysisSimulation(
    grid_size=8,
    noise_level=0.02
)

wavefront_engine = WavefrontReconstructionEngine(
    grid_size=8
)

turbulence_engine = TurbulenceEstimationEngine()

dm_engine = DMTranslationEngine(
    actuator_grid=8
)

############################################################
# Shared Data
############################################################

latest_data = {
    "frame": 0,
    "shift_data": [],
    "wavefront": [],
    "zernike": {},
    "turbulence": {},
    "dm_commands": {},
    "prediction": None  # Added field for prediction output
}

############################################################
# Continuous AO Pipeline
############################################################

def adaptive_optics_pipeline():
    global latest_data
    frame = 0
    # csv_path = "data/synthetic/zernike_timeseries.csv"
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
        wavefront, zernike = wavefront_engine.process(
            shift_data
        )

        ####################################################
        # 1. Update CSV (Rolling Window)
        ####################################################
        try:
            if os.path.exists(csv_path):
                # Read the existing CSV
                df = pd.read_csv(csv_path)
                
                # Construct the new row based on the zernike output
                new_row = {
                    "frame": frame,
                    "z1": zernike.get("z1", 0.0),
                    "z2": zernike.get("z2", 0.0),
                    "z3": zernike.get("z3", 0.0),
                    "z4": zernike.get("z4", 0.0),
                    "z5": zernike.get("z5", 0.0),
                    "z6": zernike.get("z6", 0.0)
                }
                
                # Append new row and drop the oldest row (index 0) to keep length constant
                df = pd.concat([df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
                
                # Save it back out
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error updating CSV: {e}")

        ####################################################
        # 2. Run Prediction
        ####################################################
        try:
            prediction_output = run_prediction()
        except Exception as e:
            print(f"Error running prediction: {e}")
            prediction_output = {"error": str(e)}

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
        dm_commands = dm_engine.process(
            wavefront,
            zernike
        )

        ####################################################
        # Store Latest Results
        ####################################################
        latest_data = {
            "frame": frame,
            "shift_data": shift_data,
            "wavefront": wavefront.tolist(),
            "zernike": zernike,
            "turbulence": turbulence,
            "dm_commands": dm_commands,
            "prediction": prediction_output  # Include prediction in the API payload
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
    return jsonify(latest_data)

############################################################
# Start Background Thread
############################################################

Thread(
    target=adaptive_optics_pipeline,
    daemon=True
).start()

############################################################

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )