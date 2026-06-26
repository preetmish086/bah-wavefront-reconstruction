import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import ZernikeDataset
from model import ZernikeLSTM

from config import (
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "synthetic" / "zernike_timeseries.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.pth"

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


def train_model(csv_path=CSV_PATH):

    dataset = ZernikeDataset(
        csv_path,
        sequence_length=SEQUENCE_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ZernikeLSTM()

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()


    for epoch in range(EPOCHS):

        epoch_loss = 0

        for X_batch, y_batch in dataloader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.6f}"
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nModel Saved.")
    print(
        f"Final Training Loss: {avg_loss:.6f}"
    )

if __name__ == "__main__":

    train_model()