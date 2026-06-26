import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import ZernikeLSTM
from sklearn.metrics import mean_squared_error
from config import SEQUENCE_LENGTH


CSV_PATH = "data/synthetic/zernike_timeseries.csv"
MODEL_PATH = "models/lstm_model.pth"


df = pd.read_csv(CSV_PATH)

data = df.iloc[:, 1:].values.astype("float32")

test_indices = [
    700,
    750,
    800,
    850,
    900
]

model=ZernikeLSTM()

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

all_mse = []
all_mae = []

for test_index in test_indices:

    input_sequence = data[
        test_index:
        test_index + SEQUENCE_LENGTH
    ]

    actual = data[
        test_index + SEQUENCE_LENGTH
    ]

    input_tensor = torch.tensor(
        input_sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        predicted = model(
            input_tensor
        ).squeeze().numpy()

    mse = mean_squared_error(
        actual,
        predicted
    )

    mae = np.mean(
        np.abs(actual - predicted)
    )

    all_mse.append(mse)
    all_mae.append(mae)

    print(
        f"\nIndex: {test_index}"
    )

    print(
        f"MSE: {mse:.6f}"
    )

    print(
        f"MAE: {mae:.6f}"
    )

print("\nAverage Results")

print(
    f"Average MSE: "
    f"{np.mean(all_mse):.6f}"
)

print(
    f"Average MAE: "
    f"{np.mean(all_mae):.6f}"
)
