# ML for Extreme Weather Event Prediction

This repository is a **starter, reproducible workflow** for building and evaluating machine-learning baselines for **extreme climate / weather event prediction** (rare-event classification), with parallel implementations in:

- **Python** (`src/`)
- **R** (`r/`)
- **Jupyter Notebook** (`notebooks/`)

The goal is to provide a clean project skeleton you can adapt to real datasets (reanalysis, station data, gridded climate products) while keeping the evaluation aligned with what matters for extremes: **class imbalance, calibration, thresholding, and event-focused metrics**.

## What this project does

- **Creates a simple “extreme event” label** via a user-configurable percentile threshold (e.g., \(y=1\) if target exceeds the 95th percentile).
- **Splits data with leakage-aware logic** (time-ordered split option supported in the notebook; scripts default to a randomized split for the synthetic demo).
- Trains a small baseline suite:
  - **Logistic Regression** (interpretable baseline)
  - **Random Forest** (non-linear baseline)
- Reports evaluation targeted to rare events:
  - **Precision / Recall / F1**
  - **ROC-AUC** (with imbalance caveats)
  - **PR-AUC** (preferred for rare events)
  - **Brier score** (probabilistic calibration)
- Saves light outputs (figures + metrics CSV) into `outputs/` (ignored by git by default).

## Repository structure

```
ML for Extreme Weather Event Prediction/
  README.md
  requirements.txt
  LICENSE
  .gitignore
  .gitattributes
  src/
    extreme_event_prediction.py
  r/
    extreme_event_prediction.R
  notebooks/
    extreme_event_prediction_workflow.ipynb
  data/
    .gitkeep
  outputs/
    .gitkeep
```

## Quickstart (Python)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the pipeline

```bash
python -m src.extreme_event_prediction
```

Outputs (metrics + plots) will be written to `outputs/`.

## Quickstart (Notebook)

```bash
jupyter notebook
```

Then open `notebooks/extreme_event_prediction_workflow.ipynb` and run cells top-to-bottom.

## Quickstart (R)

Open R (or RStudio) in the repo root and run:

```r
source("r/extreme_event_prediction.R")
```

The R script is self-contained for the synthetic demo and will write outputs to `outputs/`.

## Using your own climate dataset

You can replace the synthetic generator with your own dataset by producing a tabular dataframe with:

- **Predictors**: meteorological / climate covariates (and/or lagged values)
- **Target**: a continuous quantity from which extremes are defined (e.g., daily max temperature, precipitation accumulation, wind gust)
- **Time column** (strongly recommended): for leakage control and realistic validation

Suggested dataset sources (examples):

- Reanalysis: ERA5
- Satellite / gridded precipitation: CHIRPS, IMERG
- Station datasets / national services (where licensing permits)

## Notes on best practices for extremes

- **Leakage control** matters more than model choice. Prefer blocked time splits, rolling origin validation, or spatial cross-validation depending on your task.
- **PR-AUC and calibration** are often more informative than accuracy for rare events.
- If your downstream use is “issue a warning”, calibrate probabilities and choose thresholds with stakeholder costs in mind (false negatives vs false positives).

## Excluded local files (not committed)

By instruction, these files are present locally but **intentionally excluded from git history** via `.gitignore`:

- `Guidelines_Research_Paper_Review.txt`
- `Extreme_Event_Prediction_blog_post.md`

## License

MIT (see `LICENSE`).
