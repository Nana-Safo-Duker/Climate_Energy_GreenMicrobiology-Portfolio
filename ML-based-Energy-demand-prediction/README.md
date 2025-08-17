# ML-Based Energy Demand Prediction

A comprehensive machine learning project for predicting energy demand with reproducible Python, R, and notebook workflows. It supports synthetic startup data and provides a straightforward path to real utility-scale time-series forecasting.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Data Requirements
- Models and Evaluation
- Outputs
- License

## Overview

This repository is designed to demonstrate practical energy demand forecasting under time-series constraints. It includes baseline model workflows, reproducible preprocessing patterns, and dual-language implementations for easier cross-validation of results.

## Features

- Python and R demand prediction scripts
- Jupyter notebook for interactive analysis
- Chronological train/test strategy to avoid leakage
- Lag and rolling feature engineering patterns
- Modular setup for replacing synthetic data with real data

## Project Structure

```text
.
├── notebooks/
│   └── energy_demand_prediction.ipynb
├── r/
│   └── energy_demand_prediction.R
├── src/
│   └── energy_demand_prediction.py
├── Machine_Learning_Based_Energy_Demand_Prediction_Blog_Post.md
├── Guidelines_Research_Paper_Review.txt
├── Guidelines_Research_Paper_Review.md
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

R setup:

```r
install.packages(c("ggplot2","dplyr","lubridate","tidyr","zoo"))
```

## Usage

### Python

```bash
python src/energy_demand_prediction.py --model ridge
python src/energy_demand_prediction.py --model hgb
```

### R

```bash
Rscript r/energy_demand_prediction.R
```

### Notebook

```bash
jupyter notebook notebooks/energy_demand_prediction.ipynb
```

## Data Requirements

Default workflows generate synthetic hourly data. For real deployment, replace data loading with your dataset and preserve:

- chronological splitting
- lag/rolling feature computation from past values only
- explicit handling of seasonality and missing periods

## Models and Evaluation

- Baseline linear and tree-based demand models
- Metrics centered on forecasting error (e.g., MAE, RMSE)
- Time-aware evaluation for realistic performance estimates

## Outputs

Script and notebook runs produce printed metrics and optional visual outputs for model comparison and reporting.

## License

MIT License. See `LICENSE`.

