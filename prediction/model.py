import torch
import torch.nn as nn
from config import (
    NUM_ZERNIKE_MODES,
    HIDDEN_SIZE,
    NUM_LAYERS
)


class ZernikeLSTM(nn.Module):
    def __init__(
        self,
        input_size=NUM_ZERNIKE_MODES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=NUM_ZERNIKE_MODES
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output