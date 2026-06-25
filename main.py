from flask import Flask, jsonify, render_template
from threading import Thread
import time

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

    "dm_commands": {}
}

############################################################
# Continuous AO Pipeline
############################################################

def adaptive_optics_pipeline():

    global latest_data

    frame = 0

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

            "dm_commands": dm_commands
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