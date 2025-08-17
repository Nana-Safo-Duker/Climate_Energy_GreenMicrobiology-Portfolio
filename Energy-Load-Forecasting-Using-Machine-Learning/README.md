# Energy Load Forecasting Using Machine Learning

A comprehensive machine learning project for short-term energy load forecasting with reproducible Python and R workflows. The repository supports synthetic or real datasets, time-series-safe feature engineering, and robust model evaluation.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Dataset Format
- Modeling Approach
- Outputs
- Extensions
- License

## Overview

This project provides an end-to-end forecasting scaffold for energy load prediction, including scripts and a notebook for baseline modeling, evaluation, and reporting. It is suitable for paper-review assignments, coursework, and portfolio demonstrations.

## Features

- Python and R implementations for cross-language reproducibility
- Time-series-aware evaluation (no random leakage)
- Feature engineering for cyclical and autoregressive effects
- Notebook and CLI support
- Easy switch from synthetic demo data to operational load data

## Project Structure

```text
.
├── notebooks/
│   └── 01_energy_load_forecasting_ml.ipynb
├── src/
│   ├── energy_load_forecasting.py
│   └── energy_load_forecasting.R
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

R package setup (if needed):

```r
install.packages(c("dplyr","lubridate","ggplot2","randomForest","zoo","tidyr"))
```

## Usage

### Python

```bash
python src/energy_load_forecasting.py
python src/energy_load_forecasting.py --csv path\to\your_data.csv
```

### R

```bash
Rscript src/energy_load_forecasting.R
Rscript src/energy_load_forecasting.R path\to\your_data.csv
```

### Notebook

```bash
jupyter notebook notebooks/01_energy_load_forecasting_ml.ipynb
```

## Dataset Format

Minimum required columns:

- `datetime`
- `load`

Optional exogenous columns:

- `temp_c` and other weather or calendar drivers

## Modeling Approach

- Features: hour/day/month, cyclical encodings, lag terms, rolling windows
- Models: linear baseline + tree-based baseline
- Metrics: MAE, RMSE, MAPE
- Validation: chronological CV and holdout splits

## Outputs

Generated outputs include model metrics and plots from scripts or notebook runs, typically written to local working directories or configured output paths.

## Extensions

- Add ARIMA/Prophet/boosting baselines
- Add probabilistic forecasts and intervals
- Add horizon-specific backtesting
- Add drift monitoring and retraining logic

## License

MIT License. See `LICENSE`.

**Last Updated**: August 2025
