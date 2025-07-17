# Forecasting Electricity Price Index with Machine Learning

A comprehensive electricity price forecasting project with reproducible Python and R pipelines, notebook-based analysis, and time-aware model evaluation strategies for real-world energy market data.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Dataset
- Modeling Strategy
- Outputs
- Extensions
- License

## Overview

This project provides a practical framework for forecasting electricity price indices from historical market and exogenous variables. It emphasizes reproducibility, fair model comparison, and chronological validation.

## Features

- Python and R forecasting implementations
- Notebook for exploratory analysis and reproducible reporting
- Lag and rolling feature engineering patterns
- Time-series-aware train/validation methodology
- Support for external predictors (weather, load, fuel, market regime)

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
│   └── electricity_price_forecasting_workflow.ipynb
├── r/
│   └── electricity_price_forecasting_pipeline.R
├── src/
│   └── electricity_price_forecasting_pipeline.py
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

R packages:

```r
install.packages(c("tidyverse", "lubridate", "rsample", "yardstick", "slider", "ranger"))
```

## Usage

### Python

```bash
python src/electricity_price_forecasting_pipeline.py
```

### R

```r
source("r/electricity_price_forecasting_pipeline.R")
```

### Notebook

```bash
jupyter notebook notebooks/electricity_price_forecasting_workflow.ipynb
```

## Dataset

Place your dataset at `data/raw/electricity_price_index.csv`.

Minimum columns:

- `timestamp`
- `price_index`

Recommended exogenous features:

- demand/load
- weather variables
- fuel prices
- market regime indicators

## Modeling Strategy

- Feature engineering with lagged and rolling signals
- Baseline and non-linear model comparison
- Chronological splits for honest performance estimation
- Metric reporting tailored to forecasting accuracy

## Outputs

Pipelines produce local artifacts such as evaluation metrics, forecast comparison plots, and processed tables suitable for paper-review reporting.

## Extensions

- Add boosting and deep sequence models
- Add probabilistic forecasts and interval calibration
- Add concept drift monitoring across market regimes
- Add experiment tracking and config management

## License

MIT License. See `LICENSE`.
