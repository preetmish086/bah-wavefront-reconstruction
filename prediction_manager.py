from predict import predict_next_zernike
from train import train_model
from config import RETRAIN_INTERVAL


prediction_counter = 0

training_status = {
    "training": False
}


def run_prediction():

    global prediction_counter

    predicted_coeffs = predict_next_zernike()

    prediction_counter += 1

    if prediction_counter > RETRAIN_INTERVAL:

        training_status["training"] = True

        print("\nRetraining LSTM Model...\n")

        train_model()

        training_status["training"] = False

        prediction_counter = 0

        predicted_coeffs = predict_next_zernike()

    return {
        "prediction": predicted_coeffs,
        "training": training_status["training"]
    }