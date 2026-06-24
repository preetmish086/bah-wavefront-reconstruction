import torch
from model import ZernikeLSTM
model=ZernikeLSTM()

dummy_input=torch.randn(
    32,
    20,
    5
)

output=model(dummy_input)

print("Output Shape:", output.shape)