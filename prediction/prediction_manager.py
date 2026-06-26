from predict import predict_next_zernike
from train import train_model
from config import RETRAIN_INTERVAL


prediction_counter = 0


def run_prediction():

    global prediction_counter

    predicted_coeffs = predict_next_zernike()

    prediction_counter += 1

    if prediction_counter > RETRAIN_INTERVAL:

        print("\nRetraining LSTM Model...\n")

        train_model()

        prediction_counter = 0

    return predicted_coeffs