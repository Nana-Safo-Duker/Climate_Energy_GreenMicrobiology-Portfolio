# Climate Data for Renewable Energy Optimization

A comprehensive machine learning project for renewable power forecasting with time-series-aware optimization. The repository provides Python, R, and notebook workflows to compare baseline and tuned models using rolling backtests and reproducible reporting.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Data Format
- Methodology
- Outputs
- Reproducibility
- License

## Overview

This project demonstrates data-driven optimization for renewable forecasting by evaluating a baseline model against an optimized alternative under strict chronological validation. It is built for reproducible experimentation and easy extension to real operational datasets.

## Features

- Baseline vs optimized model comparison
- Rolling/expanding window backtesting
- Dual-language implementation (`Python` + `R`)
- Notebook workflow for interactive analysis
- Auto-generated synthetic dataset for no-friction startup
- Exported metrics and figures for reporting

## Project Structure

```text
.
├── data/
│   └── .gitkeep
├── notebooks/
│   └── 01_forecasting_pipeline.ipynb
├── reports/
│   ├── .gitkeep
│   └── figures/
│       └── .gitkeep
├── src/
│   ├── forecasting_pipeline.py
│   └── forecasting_pipeline.R
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

### Python

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### R

```r
install.packages(c(
  "tidyverse","lubridate","slider","recipes","parsnip",
  "workflows","rsample","yardstick","glmnet","ranger"
))
```

## Usage

### Python pipeline

```bash
python src/forecasting_pipeline.py
```

### R pipeline

```bash
Rscript src/forecasting_pipeline.R
```

### Notebook

```bash
jupyter notebook notebooks/01_forecasting_pipeline.ipynb
```

## Data Format

Map your dataset to the following columns:

- `timestamp`: datetime index
- `renewable_power`: target variable
- feature columns: meteorological and operational drivers

Suggested examples: `irradiance`, `wind_speed`, `temperature_c`, `humidity_pct`.

## Methodology

- Baseline model: fast linear benchmark (e.g., Ridge)
- Optimized model: tree-based model with small grid search
- Validation: rolling time-series cross-validation only
- Metrics: MAE-focused comparison with fold-level reporting

## Outputs

Typical generated artifacts:

- `reports/rolling_backtest_metrics.csv`
- `reports/rolling_backtest_metrics_R.csv`
- `reports/figures/rolling_backtest_mae.png`
- `reports/figures/rolling_backtest_mae_R.png`

## Reproducibility

- Avoid leakage by preserving chronological splits.
- Document search space and CV settings.
- Report fold variability, not only aggregated means.
- Keep generated artifacts local unless explicitly needed.

## License

MIT License. See `LICENSE`.

**Last Updated**: July 2025
