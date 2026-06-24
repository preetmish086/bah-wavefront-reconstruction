import torch
import torch.nn as nn


class ZernikeLSTM(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_layers=2,
        output_size=5
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

        lstm_out, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output