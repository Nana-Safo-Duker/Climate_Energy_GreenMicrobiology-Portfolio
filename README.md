# Forecasting Electricity Price Index with Machine Learning Models and Strategies

This repository contains a complete, reproducible project starter for reviewing and extending research on electricity price index forecasting using machine learning.

It includes:
- Python and R forecasting pipelines,
- a Jupyter notebook workflow,
- standard GitHub repository setup files.

## Project Goals

1. Translate a research paper into a rigorous, readable scientific blog post.
2. Build reproducible forecasting workflows in Python and R.
3. Compare machine learning models under time-aware validation.
4. Document results clearly for future academic or professional use.

## Repository Structure

```text
.
|-- .gitattributes
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   `-- electricity_price_forecasting_workflow.ipynb
|-- r/
|   `-- electricity_price_forecasting_pipeline.R
`-- src/
    `-- electricity_price_forecasting_pipeline.py
```

## Files Overview

- `src/electricity_price_forecasting_pipeline.py`  
  End-to-end Python pipeline with lag/rolling feature engineering, model training, and cross-validated performance reporting.

- `r/electricity_price_forecasting_pipeline.R`  
  Equivalent R workflow using tidyverse + time-aware resampling with baseline and tree-based models.

- `notebooks/electricity_price_forecasting_workflow.ipynb`  
  Interactive notebook for EDA, feature creation, CV model comparison, plots, and interpretation notes.

## Data Requirements

Create your dataset at:

`data/raw/electricity_price_index.csv`

Minimum required columns:

- `timestamp` (datetime-compatible)
- `price_index` (numeric target variable)

Optional exogenous columns (recommended):

- demand/load variables
- weather features (temperature, wind, solar)
- fuel price indicators
- market regime flags

## Quick Start (Python)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your dataset to `data/raw/electricity_price_index.csv`.
4. Run:

```bash
python src/electricity_price_forecasting_pipeline.py
```

By default, any locally generated outputs should go to an ignored folder (e.g. `outputs/`) so the repo stays clean.

## Quick Start (Jupyter Notebook)

```bash
jupyter notebook notebooks/electricity_price_forecasting_workflow.ipynb
```

Run all cells after preparing the CSV dataset.

## Quick Start (R)

Install required R packages if needed:

```r
install.packages(c(
  "tidyverse", "lubridate", "rsample", "yardstick", "slider", "ranger"
))
```

Run:

```r
source("r/electricity_price_forecasting_pipeline.R")
```

## Methodological Notes

- Uses **time-series-aware validation** (not random train/test split).
- Compares multiple model families to avoid single-model bias.
- Emphasizes reproducibility and transparent metric reporting.
- Includes descriptive and inferential framing aligned with scientific review practice.

## Suggested Extensions

1. Add XGBoost, LightGBM, CatBoost, and sequence models (LSTM/Temporal CNN).
2. Evaluate probabilistic forecasts (prediction intervals, quantile loss).
3. Add exogenous features and feature importance diagnostics.
4. Perform drift analysis across market regimes and seasons.
5. Add experiment tracking (MLflow or lightweight logging).

## Reproducibility and Ethics

- Keep raw data immutable and log preprocessing choices.
- Record versions of datasets, packages, and model configurations.
- Report limitations and uncertainty, especially for high-impact decisions.
- Avoid overclaiming causal insights from predictive models.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgment

Primary reviewed paper:

*Forecasting Electricity Price Index with Machine Learning Models and Strategies*  
ResearchGate link: [https://www.researchgate.net/publication/390288615_Forecasting_Electricity_Price_Index_with_Machine_Learning_Models_and_Strategies](https://www.researchgate.net/publication/390288615_Forecasting_Electricity_Price_Index_with_Machine_Learning_Models_and_Strategies)
