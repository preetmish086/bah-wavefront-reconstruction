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


os.makedirs(
    "models",
    exist_ok=True
)


CSV_PATH = "data/synthetic/zernike_timeseries.csv"



dataset = ZernikeDataset(
    csv_path=CSV_PATH,
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
    "models/lstm_model.pth"
)

print("\nModel Saved.")
print(
    f"Final Training Loss: {avg_loss:.6f}"
)