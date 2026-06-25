# Prediction Engine

Forecasts future Zernike coefficient vectors using historical coefficient sequences.

Current implementation:
- Synthetic turbulence-inspired dataset
- LSTM forecasting model
- Next-frame prediction
- Evaluation using MSE and MAE

Future:
- Replace synthetic data with reconstructed Zernike coefficients from SH-WFS pipeline
- Add autoregressive baseline comparison