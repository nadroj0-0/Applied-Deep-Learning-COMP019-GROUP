# M5 Forecasting: Hierarchical Deep Learning Pipeline

This repository contains a modular framework for hierarchical time-series forecasting using the M5 Competition dataset. The project evaluates multiple architectures, including traditional baseline LSTMs, Temporal Fusion Transformers (TFT), and custom Hierarchical Gated Recurrent Units (GRU).

## Project Overview

The core of this project is a "Plug-and-Play" architecture where every model inherits from a standard base class. This ensures that data splitting, evaluation metrics (WSPL, CRPS, R²), and visualization are consistent across all experiments.

### Model Performance Summary

Based on our evaluation metrics, the models are ranked by their Weighted Scaled Pinball Loss (WSPL) and R² accuracy:

| Model | WSPL | R² | Coverage (95%) | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Hierarchical LSTM** | **0.398** | **0.694** | 0.974 | 2.013 |
| **Hierarchical Quantile GRU** | 0.410 | 0.696 | 0.955 | 2.005 |
| **Regular GRU** | 0.404 | 0.686 | 0.971 | 2.039 |
| **Hierarchical Probabilistic GRU** | 0.458 | 0.680 | 0.982 | 2.057 |
| **LightGBM NN Hybrid** | 0.466 | 0.411 | 0.978 | 2.792 |
| **LSTM Baseline** | 0.511 | 0.450 | 0.939 | 2.699 |
| **TFT** | 0.561 | 0.580 | 0.950 | 2.358 |

**Top Performer:** The **Hierarchical LSTM** provided the best balance of point accuracy and uncertainty calibration, achieving the lowest WSPL.

## Visualizing Forecasts

The following "Fan Plots" illustrate the predicted median sales against actual sales, including the 50% and 95% confidence intervals.

### Hierarchical LSTM Forecast
![Hierarchical LSTM Forecast](https://github.com/nadroj0-0/Applied-Deep-Learning-COMP019-GROUP/blob/main/outputs/hierarchical_lstm/forecast_plot_FOODS_3_120_CA_3_evaluation.png?raw=true)

### Temporal Fusion Transformer (TFT) Forecast
![TFT Forecast](https://github.com/nadroj0-0/Applied-Deep-Learning-COMP019-GROUP/blob/main/outputs/tft/forecast_plot_FOODS_3_120_CA_3_evaluation.png?raw=true)

---

## Installation and Setup

1. **Environment:** Ensure your environment meets the dependencies listed in requirements.txt.
2. **Install:**
   pip install -r requirements.txt
3. **Execute:**
   python run_pipeline.py

## System Requirements and Performance

This pipeline is optimized for computational throughput rather than memory conservation. To ensure stable execution:

* **Memory:** A minimum of 16GB System RAM is required.
* **Storage:** Sufficient disk space for raw data and cached .pkl objects.
* **Runtime:** Initial data processing typically requires 4 to 5 minutes, with a total pipeline run time of approximately 7 minutes (excluding deep learning training phases).

## Evaluation Framework

Models are evaluated primarily on the Weighted Scaled Pinball Loss (WSPL) to assess the quality of the probabilistic uncertainty distribution. Key metrics include:

* **Coverage Error:** Validates the capture rate of the 95% confidence interval.
* **R² Score:** Assesses variance capture, with a success benchmark of > 0.65.
* **Quantile Fan Plots:** Automatically generated and saved in model-specific subdirectories within the outputs/ directory.

The consolidated performance results are exported to model_comparison.csv.

## Repository Architecture

* **Root Directory:** Contains the main pipeline, core model classes, and evaluation logic.
* **outputs/:** Stores generated plots, weights, and compressed .csv.gz predictions.
* **ablation-study/:** Contains comparative research code specifically for the TFT and GRU architecture experiments.
