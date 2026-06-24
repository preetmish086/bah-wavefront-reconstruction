import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class ZernikeDataset(Dataset):
    def __init__(
        self,
        csv_path,
        sequence_length=20
    ):
        self.sequence_length = sequence_length

        df = pd.read_csv(csv_path)

        self.data = df.drop(
            columns=["frame"]
        ).values.astype(np.float32)

        self.X = []
        self.y = []

        for i in range(
            len(self.data) - sequence_length
        ):
            self.X.append(
                self.data[i:i + sequence_length]
            )

            self.y.append(
                self.data[i + sequence_length]
            )

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx])
        )