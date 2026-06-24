from dataset import ZernikeDataset

dataset = ZernikeDataset(
    csv_path="data/synthetic/zernike_timeseries.csv",
    sequence_length=20
)

print("Dataset Size:", len(dataset))

X, y = dataset[0]

print("Input Shape:", X.shape)
print("Target Shape:", y.shape)