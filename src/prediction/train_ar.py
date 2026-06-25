import pandas as pd
import joblib

from ar_model import ZernikeAR


CSV_PATH = "data/synthetic/zernike_timeseries.csv"

df = pd.read_csv(CSV_PATH)

data = df.iloc[:, 1:].values.astype("float32")

model = ZernikeAR()

model.train(data)

joblib.dump(
    model,
    "models/ar_model.pkl"
)

print("AR Model Saved.")