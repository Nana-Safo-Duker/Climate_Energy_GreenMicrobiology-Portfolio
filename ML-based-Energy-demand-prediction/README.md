# Machine learning based energy demand prediction (reproducible repo)

This folder contains:

- a **reproducible Python + R workflow** demonstrating time-series-safe feature engineering and evaluation for energy-demand prediction
- retained placeholder files for excluded guideline/blog content

## Contents

- `Machine_Learning_Based_Energy_Demand_Prediction_Blog_Post.md`: retained file, content excluded
- `Guidelines_Research_Paper_Review.txt`: retained file, content excluded
- `Guidelines_Research_Paper_Review.md`: retained file, content excluded
- `src/energy_demand_prediction.py`: runnable Python demo (prints MAE/RMSE)
- `notebooks/energy_demand_prediction.ipynb`: detailed notebook workflow
- `r/energy_demand_prediction.R`: runnable R demo (prints MAE/RMSE + plots)
- `.gitignore`, `.gitattributes`, `LICENSE`, `requirements.txt`: repository scaffolding

## Quickstart (Python)

### 1) Create and activate a virtual environment (recommended)

PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Run the script

```bash
python .\src\energy_demand_prediction.py --model ridge
python .\src\energy_demand_prediction.py --model hgb
```

What you’ll see:

- train/test sizes (time split)
- MAE and RMSE in MW

## Quickstart (Notebook)

After installing dependencies:

```bash
jupyter notebook
```

Then open `notebooks/energy_demand_prediction.ipynb`.

## Quickstart (R)

### 1) Confirm R is available

```bash
Rscript --version
```

### 2) Install R packages (one-time)

In an R console:

```r
install.packages(c("ggplot2","dplyr","lubridate","tidyr","zoo"))
```

### 3) Run the R script

```bash
Rscript .\r\energy_demand_prediction.R
```

## Notes on data and reproducibility

- The Python and R workflows use a **synthetic but realistic hourly dataset** so the repo runs without credentials or private utility data.
- The workflow is designed to be **time-series safe**:
  - lag and rolling features are computed using `shift()/lag()` so they only use the past
  - the train/test split is chronological (no shuffling)

To use a real dataset, replace the synthetic data generator with your dataset loader and keep the same feature and splitting principles.

## License

This project is licensed under the MIT License (see `LICENSE`).

