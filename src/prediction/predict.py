import torch
import pandas as pd

from model import ZernikeLSTM

from config import SEQUENCE_LENGTH

CSV_PATH = "data/synthetic/zernike_timeseries.csv"
MODEL_PATH = "models/lstm_model.pth"


df = pd.read_csv(CSV_PATH)

data = df.iloc[:, 1:].values.astype("float32")


input_sequence = data[-SEQUENCE_LENGTH:]


input_tensor = torch.tensor(
    input_sequence,
    dtype=torch.float32
).unsqueeze(0)

model = ZernikeLSTM()

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

with torch.no_grad():

    prediction = model(
        input_tensor
    )

predicted_coeffs = (
    prediction
    .squeeze()
    .numpy()
)

print("\nPredicted Next Zernike Vector:\n")

for i, value in enumerate(
    predicted_coeffs,
    start=1
):
    print(
        f"z{i}: {value:.4f}"
    )