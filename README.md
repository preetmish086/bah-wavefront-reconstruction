# BAH Adaptive Optics

## Overview

This project is a collaborative Minimum Viable Product (MVP) designed to advance adaptive optics technology. It provides a comprehensive pipeline for atmospheric turbulence analysis, Zernike coefficient prediction and wavefront reconstruction.
The primary goal is to predict future Zernike coefficients to reconstruct future wavefronts, allowing adaptive optics systems to compensate for atmospheric distortions more effectively.

## 🚀 Key Features

*   **Atmospheric Analysis:** Advanced tools designed for the detailed study of atmospheric turbulence patterns that can degrade optical signals.
*   **Predictive Modelling:** 
    *   **LSTM Forecasting:** Utilises Long Short-Term Memory (LSTM) networks for high-precision time-series prediction of wavefront variations.
    *   **Autoregressive (AR) Models:** Features integrated AR model implementations for robust, statistical forecasting of Zernike coefficients.
*   **Wavefront Reconstruction:** Seamlessly translates predicted Zernike coefficients back into high-fidelity reconstructed wavefronts for real-time adaptive optics correction.
*   **Web Portal:** A comprehensive, built-in application providing a streamlined interface for managing model predictions and visualising results.
*   **Data Pipeline:** A robust suite of scripts for generating synthetic turbulence data, visualising complex datasets, and conducting thorough model performance evaluations.

## Project Structure

- data/ : datasets and generated data
- results/ : outputs, plots and model results
- src/ : source code (to be added)
- notebooks/ : experiments and analysis notebooks (to be added)

## MVP Goal

Predict future Zernike coefficients and reconstruct future wavefronts for adaptive optics applications.
