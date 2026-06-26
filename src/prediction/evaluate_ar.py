import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

from config import SEQUENCE_LENGTH


CSV_PATH = "data/synthetic/zernike_timeseries.csv"

df = pd.read_csv(CSV_PATH)

data = df.iloc[:, 1:].values.astype("float32")

model = joblib.load(
    "models/ar_model.pkl"
)

test_indices = [
    700,
    750,
    800,
    850,
    900
]

all_mse = []

all_mae = []

for test_index in test_indices:

    sequence = data[
        test_index:
        test_index + SEQUENCE_LENGTH
    ]

    actual = data[
        test_index + SEQUENCE_LENGTH
    ]

    predicted = model.predict(
        sequence
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    mae = np.mean(
        np.abs(
            actual - predicted
        )
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