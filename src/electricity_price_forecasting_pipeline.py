"""
Electricity Price Index Forecasting Pipeline (Python)
----------------------------------------------------
This script provides a reproducible starter pipeline for:
1) loading and validating electricity price time-series data,
2) feature engineering with lag/rolling/calendar variables,
3) training ML baselines and advanced models,
4) evaluating forecasts with common metrics,
5) saving artifacts for downstream analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    data_path: Path = Path("data/raw/electricity_price_index.csv")
    timestamp_col: str = "timestamp"
    target_col: str = "price_index"
    output_dir: Path = Path("outputs")
    n_splits: int = 5
    random_state: int = 42


def ensure_directories(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def load_data(cfg: Config) -> pd.DataFrame:
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Input data not found at {cfg.data_path}. "
            "Create the CSV with at least [timestamp, price_index] columns."
        )

    df = pd.read_csv(cfg.data_path)
    if (
        cfg.timestamp_col not in df.columns
        or cfg.target_col not in df.columns
    ):
        raise ValueError(
            "Missing required columns. "
            f"Needed: {cfg.timestamp_col}, {cfg.target_col}"
        )

    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df = df.dropna(
        subset=[cfg.timestamp_col, cfg.target_col]
    ).sort_values(cfg.timestamp_col)
    df = df.reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[timestamp_col].dt.hour
    out["dayofweek"] = out[timestamp_col].dt.dayofweek
    out["month"] = out[timestamp_col].dt.month
    out["quarter"] = out[timestamp_col].dt.quarter
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def add_lag_rolling_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    for window in [3, 6, 12, 24]:
        out[f"roll_mean_{window}"] = out[target_col].rolling(
            window=window
        ).mean()
        out[f"roll_std_{window}"] = out[target_col].rolling(
            window=window
        ).std()

    return out


def preprocess_features(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, pd.Series]:
    work = add_time_features(df, cfg.timestamp_col)
    work = add_lag_rolling_features(work, cfg.target_col)

    # Keep rows where lag/rolling features are available.
    work = work.dropna().reset_index(drop=True)

    feature_cols = [
        c for c in work.columns
        if c not in [cfg.timestamp_col, cfg.target_col]
    ]
    x = work[feature_cols]
    y = work[cfg.target_col]
    return x, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_time_series_cv(
    x: pd.DataFrame, y: pd.Series, cfg: Config
) -> Dict[str, List[Dict[str, float]]]:
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)

    models = {
        "LinearRegression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
    }

    metrics_by_model: Dict[str, List[Dict[str, float]]] = {
        k: [] for k in models
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            fold_metrics = evaluate(y_test.to_numpy(), pred)
            fold_metrics["fold"] = fold
            metrics_by_model[model_name].append(fold_metrics)

    return metrics_by_model


def summarize_cv(
    metrics_by_model: Dict[str, List[Dict[str, float]]]
) -> pd.DataFrame:
    rows = []
    for model_name, records in metrics_by_model.items():
        df_m = pd.DataFrame(records)
        rows.append(
            {
                "model": model_name,
                "MAE_mean": df_m["MAE"].mean(),
                "MAE_std": df_m["MAE"].std(),
                "RMSE_mean": df_m["RMSE"].mean(),
                "RMSE_std": df_m["RMSE"].std(),
                "R2_mean": df_m["R2"].mean(),
                "R2_std": df_m["R2"].std(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("RMSE_mean")
    return summary


def save_outputs(
    metrics_by_model: Dict[str, List[Dict[str, float]]],
    summary_df: pd.DataFrame,
    cfg: Config,
) -> None:
    raw_metrics_path = cfg.output_dir / "cv_metrics.json"
    summary_path = cfg.output_dir / "cv_summary.csv"

    with raw_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, indent=2)

    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {raw_metrics_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    cfg = Config()
    ensure_directories(cfg)

    print("Loading data...")
    df = load_data(cfg)
    print(f"Loaded {len(df):,} rows.")

    print("Creating features...")
    x, y = preprocess_features(df, cfg)
    print(f"Feature matrix: {x.shape[0]:,} rows x {x.shape[1]:,} columns")

    print("Running cross-validation...")
    metrics_by_model = run_time_series_cv(x, y, cfg)
    summary_df = summarize_cv(metrics_by_model)

    print("\nCross-validated model summary:")
    print(summary_df.to_string(index=False))

    save_outputs(metrics_by_model, summary_df, cfg)
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
