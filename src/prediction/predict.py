import torch
import pandas as pd

from model import ZernikeLSTM
from config import SEQUENCE_LENGTH

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "synthetic" / "zernike_timeseries.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.pth"



def predict_next_zernike(sequence=None, csv_path=CSV_PATH):

    if sequence is None:

        df = pd.read_csv(csv_path)

        data = df.iloc[:, 1:].values.astype("float32")

        input_sequence = data[-SEQUENCE_LENGTH:]

    else:

        input_sequence = sequence

    input_tensor = torch.tensor(
        input_sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    model = ZernikeLSTM()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    with torch.no_grad():

        prediction = model(input_tensor)

    predicted_coeffs = (
        prediction
        .squeeze()
        .tolist()
    )

    return predicted_coeffs


if __name__ == "__main__":

    predicted_coeffs = predict_next_zernike()

    print("\nPredicted Next Zernike Vector:\n")

    for i, value in enumerate(
        predicted_coeffs,
        start=1
    ):
        print(f"Z{i}: {value:.4f}")