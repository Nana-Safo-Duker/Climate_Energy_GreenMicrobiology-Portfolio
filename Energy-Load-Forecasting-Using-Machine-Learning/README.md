# Energy Load Forecasting Using Machine Learning

This repository is a **reproducible starter project** for energy/electricity load forecasting using machine learning. It includes:
- A **scientific blog post** written using your rubric (`Energy_Load_Forecasting_Scientific_Blog_Post.md`)
- A **Python baseline pipeline** with time-series cross-validation (`src/energy_load_forecasting.py`)
- A matching **R baseline pipeline** (`src/energy_load_forecasting.R`)
- A **Jupyter notebook** that runs end-to-end (`notebooks/01_energy_load_forecasting_ml.ipynb`)

## Project structure

- `Energy_Load_Forecasting_Scientific_Blog_Post.md`: blog post following the assignment guideline
- `Guidelines_Research_Paper_Review.txt`: rubric / outline you provided
- `notebooks/01_energy_load_forecasting_ml.ipynb`: runnable notebook (synthetic data by default)
- `src/energy_load_forecasting.py`: Python pipeline (synthetic data by default)
- `src/energy_load_forecasting.R`: R pipeline (synthetic data by default)
- `.gitattributes`, `.gitignore`, `LICENSE`, `requirements.txt`: GitHub repository essentials

## Dataset format

If you have a real dataset, provide it as a CSV with at least:
- `datetime`: parseable timestamp (e.g., `2024-01-01 00:00:00`)
- `load`: numeric load value (kW/MW/etc.)

Optional but recommended:
- `temp_c`: temperature in Celsius (numeric)

Example header:

```csv
datetime,load,temp_c
2024-01-01 00:00:00,1332.5,6.2
2024-01-01 01:00:00,1288.1,5.9
```

## Quickstart (Python)

### Install

Using Python 3.10+:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the script (synthetic data)

```bash
python .\src\energy_load_forecasting.py
```

### Run the script (your CSV)

```bash
python .\src\energy_load_forecasting.py --csv path\to\your_data.csv
```

The script reports **MAE**, **RMSE**, and **MAPE** using **time-series cross-validation** (no random shuffling).

## Run the notebook

```bash
python -m ipykernel install --user --name energy-load-forecasting
jupyter notebook
```

Then open `notebooks/01_energy_load_forecasting_ml.ipynb`.

## Quickstart (R)

Install required R packages:

```r
install.packages(c("dplyr","lubridate","ggplot2","randomForest","zoo","tidyr"))
```

Run:

```bash
Rscript .\src\energy_load_forecasting.R
```

Or with a CSV:

```bash
Rscript .\src\energy_load_forecasting.R path\to\your_data.csv
```

## What’s implemented (baseline)

- **Feature engineering**
  - Calendar: hour-of-day, day-of-week, month
  - Cyclical encoding: sin/cos for hour and day-of-week
  - Autoregressive lags: \(t-1\), \(t-24\)
  - Rolling mean (24h)
- **Models**
  - Ridge regression (Python)
  - Random Forest (Python and R)
- **Evaluation**
  - `TimeSeriesSplit` in Python
  - Expanding-window CV in R (simple implementation)
  - Metrics: MAE, RMSE, MAPE (with a safe denominator)

## How to extend this project

- Add better baselines: seasonal naïve, ARIMA/ARIMAX
- Add boosting: XGBoost / LightGBM with careful time-series validation
- Add probabilistic forecasting (quantiles, prediction intervals)
- Add backtesting by horizon (1h, 24h, 7d)
- Add model monitoring (drift detection on temperature/load relationships)

## Reference (paper being reviewed)

The blog post is based on the paper linked in `Guidelines_Research_Paper_Review.txt`:
- *Energy Load Forecasting with Machine Learning Models, Metrics and Future Directions* (ResearchGate link in the prompt)

## License

MIT License. See `LICENSE`.
