import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/synthetic/zernike_timeseries.csv"
)

plt.figure(figsize=(12,6))

for col in df.columns[1:]:
    plt.plot(df[col], label=col)

plt.legend()
plt.title("Synthetic Zernike Coefficients")
plt.show()