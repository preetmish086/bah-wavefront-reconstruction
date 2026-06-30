# BAH - MVP - Team AlgoRhythmmm
> Wavefront Reconstruction, DM Control, Turbulence Characterisation and Forecasting

<a style="display:block; width: fit-content; border:0; background: #00ccff; color: #fff; padding: 10px 20px; border-radius: 50px; font-family: 'arial'; letter-spacing: 1px; text-decoration: none; margin: 25px 0; =" href="https://docs.google.com/presentation/d/1bONYWeurhWJO1tGv1b5jdY-nArSsAXbnjnjfiqRyBd4/edit?usp=sharing">SUBMITTED PPT (SYNOPSIS) ↗️</a>

## Digital Twin Web Portal

**An Intelligent Adaptive Optics Framework for Wavefront Reconstruction, Atmospheric Turbulence Estimation, Predictive Forecasting and Deformable Mirror Control**

### Main Dashboard

![WEB-DIGITAL-TWIN](assets/web.jpeg)

LIVE WORKING VIDEO - https://github.com/preetmish086/bah-wavefront-reconstruction/blob/develop/assets/web-vid.mp4

<video width="50%" controls>
  <source src="assets/web-vid.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## Overview

The Adaptive Optics Monitoring and Analysis Portal is a complete software framework that transforms raw Shack-Hartmann Wavefront Sensor (SH-WFS) images into deformable mirror correction commands while providing real-time visualization, atmospheric turbulence estimation, predictive forecasting, and system analytics.

The platform integrates the complete adaptive optics pipeline into a unified monitoring environment, enabling researchers to observe every stage of the wavefront sensing and correction process.

---

# MVP Features

- SH-WFS Image Analysis (Simulated)
- Wavefront Reconstruction
- Zernike Coefficient Extraction
- Atmospheric Turbulence Estimation
- Deformable Mirror Actuator Mapping
- LSTM-based Turbulence Prediction

---

# System Architecture and Flow Diagrams

![System Architecture](static/modal-0.png)

![System Architecture](static/modal-1.png)

![System Architecture](static/modal-2.png)
---

# INDIVIDUAL FEATURE EXPLANATION

## SH-WFS Sensor Dashboard

>**Visualize incoming wavefront sensor data and monitor spot detection performance.**

<img src="assets/image.png" style="max-width: 22rem">

* **INPUT DATA SOURCE** : SENSOR FRAME  ANALYSIS ENGINE

* **FEATURES** : Raw frame viewer, Spot Field Visualizer, Shift visualiser

ℹ️ Actual sensor frame image analysis functionalities couldn't be developed at the moment due to the absence of specified frame format and data

## Wavefront Reconstruction Dashboard
>**Visualize reconstructed atmospheric distortion and Zernike analysis**

<img src="assets/wavefront.png" style="max-width: 22rem">
<img src="assets/zernike.png" style="max-width: 22rem">

* **INPUT DATA SOURCE** : WAVEFRONT RECONSTRUCTION ENGINE

* **FEATURES** : Interactive Heatmap integrated Wavefront surface, Zernike Parameters Metrics visualizer

## Turbulence Characterization Dashboard
>**Monitor atmospheric turbulence parameters over a time series**

<img src="assets/turbulence.png" style="max-width: 22rem">

* **INPUT DATA SOURCE** : TURBULENCE ESTIMATION ENGINE

* **FEATURES** : r₀ , τ₀ time series visualizer, turbulence strength visualizer

## DM Command Dashboard
>**Visualize the the correction pattern for DM commands**

<img src="assets/dm.png" style="max-width: 22rem">

* **INPUT DATA SOURCE** : DM ACTUATOR COMMAND TRANSLATION ENGINE

* **FEATURES** : Grid magnitude based colour map, correct pattern visualisation

## Forecasting Dashboard
>**Visualize the forecasted turbulence and zernike data**

<img src="assets/predict.png" style="max-width: 22rem">

* **INPUT DATA SOURCE** : FORECASTING ENGINE

* **FEATURES** : Zernike and turbulence prediction visualisation, dm actuator predictive correction preparation


# PROJECT STRUCTURE
