import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import SEQUENCE_LENGTH


class ZernikeDataset(Dataset):
    def __init__(
        self,
        csv_path,
        sequence_length=SEQUENCE_LENGTH
    ):
        self.sequence_length = sequence_length

        df = pd.read_csv(csv_path)

        if "frame" not in df.columns:
            raise ValueError(
                "CSV must contain a 'frame' column."
            )

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
            torch.tensor(
                self.X[idx],
                dtype=torch.float32
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32
            )
        )