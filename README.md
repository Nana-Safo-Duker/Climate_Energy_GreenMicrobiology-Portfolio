## Climate Data for Renewable Energy Optimization

This repository contains:

- **A scientific blog post** reviewing *Data-Driven Optimization of Renewable Energy Forecasting* (ResearchGate publication `391526838`)
- **A reproducible forecasting workflow** (Python + R + Jupyter) that demonstrates the core idea: compare a **baseline** model against an **optimized** model under a **rolling time-series backtest**

### What’s included

- **Scientific blog post (kept locally, not pushed)**: `blog_post_research_paper_review.md`
- **Python pipeline**: `src/forecasting_pipeline.py`
- **R pipeline**: `src/forecasting_pipeline.R`
- **Notebook**: `notebooks/01_forecasting_pipeline.ipynb`
- **Outputs (generated locally; not committed)**: created under `reports/` when you run the scripts/notebook

### Project structure

- `data/`: datasets (repo ships with a `.gitkeep`; scripts will generate `synthetic_renewable_timeseries.csv` by default)
- `notebooks/`: Jupyter notebooks
- `src/`: runnable scripts
- `reports/`: metrics and figures (created by runs; kept out of git by default)

## Quickstart (Python)

### 1) Create environment and install requirements

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run the pipeline

```bash
python src/forecasting_pipeline.py
```

### 3) Check outputs

- `reports/rolling_backtest_metrics.csv`
- `reports/figures/rolling_backtest_mae.png`

## Quickstart (R)

The R script uses common tidyverse + tidymodels packages.

### 1) Install packages (example)

In R:

```r
install.packages(c(
  "tidyverse","lubridate","slider",
  "recipes","parsnip","workflows","rsample","yardstick",
  "glmnet","ranger"
))
```

### 2) Run

From the repo root:

```bash
Rscript src/forecasting_pipeline.R
```

### 3) Check outputs

- `reports/rolling_backtest_metrics_R.csv`
- `reports/figures/rolling_backtest_mae_R.png`

## Notebook usage

Open `notebooks/01_forecasting_pipeline.ipynb` and run all cells. It will:

- generate or load `data/synthetic_renewable_timeseries.csv`
- create a supervised learning table (forecast horizon + lags)
- run a rolling backtest baseline vs optimized
- save the key figure + metrics to `reports/`

## Methodology notes (what “optimized” means here)

To keep the repository runnable on a normal laptop, “optimization” is implemented as:

- **Baseline**: Ridge regression (fast, strong benchmark)
- **Optimized**: Random Forest with a small **grid search** using **time-series CV**

This matches the spirit of data-driven optimization papers: do not rely on a single hand-picked model configuration—use a systematic search with a time-respecting validation protocol.

## Replacing synthetic data with real data

If you have real renewable forecasting data, modify the scripts/notebook to load your dataset and map columns to:

- `timestamp` (datetime)
- meteorological drivers (e.g., `irradiance`, `wind_speed`, `temperature_c`, `humidity_pct`)
- `renewable_power` (the signal you want to forecast)

Then rerun the pipeline to reproduce figures on your real time series.

## Reproducibility and ethics checklist

- **No leakage**: use rolling/time-series validation, not random shuffles.
- **Document splits**: record train/test windows and horizon.
- **Report variability**: show performance across folds, not just a single average.
- **Be honest about compute**: optimization searches should be reported (grid size, CV scheme).

## License

MIT License. See `LICENSE`.

