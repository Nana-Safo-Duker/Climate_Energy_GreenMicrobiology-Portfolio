from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    mae: float
    rmse: float
    mape: float


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def make_synthetic_hourly_dataset(
    start: str = "2024-01-01",
    periods: int = 24 * 180,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Create a realistic-ish hourly load dataset with weather + calendar effects.
    Output schema matches the CSV schema expected by this project.
    """
    rng = np.random.default_rng(seed)
    dt = pd.date_range(start=start, periods=periods, freq="h")

    hour = dt.hour.to_numpy()
    dow = dt.dayofweek.to_numpy()
    doy = dt.dayofyear.to_numpy()

    # Temperature: seasonal + daily cycle + noise
    temp = (
        12
        + 10 * np.sin(2 * np.pi * (doy / 365.25))
        + 5 * np.sin(2 * np.pi * (hour / 24.0))
        + rng.normal(0, 1.2, size=periods)
    )

    # Load: base + daily + weekly + temperature effect + random noise
    base = 1200.0
    daily = 180 * np.sin(2 * np.pi * (hour / 24.0 - 0.15))
    weekly = 90 * (dow < 5).astype(float) - 60 * (dow >= 5).astype(float)
    # Nonlinear temperature effect (heating/cooling degree style)
    comfort = 18.0
    temp_effect = 15.0 * np.maximum(comfort - temp, 0) + 9.0 * np.maximum(temp - comfort, 0)
    noise = rng.normal(0, 35, size=periods)

    load = base + daily + weekly + temp_effect + noise
    load = np.maximum(load, 50.0)

    df = pd.DataFrame(
        {
            "datetime": dt,
            "load": load,
            "temp_c": temp,
        }
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])

    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    # Lag features (shifted load)
    out = out.sort_values("datetime")
    out["load_lag_1"] = out["load"].shift(1)
    out["load_lag_24"] = out["load"].shift(24)
    out["load_roll_mean_24"] = out["load"].shift(1).rolling(24).mean()

    out = out.dropna().reset_index(drop=True)
    return out


def train_and_evaluate(df: pd.DataFrame, n_splits: int = 5, seed: int = 7) -> list[ForecastResult]:
    df = add_time_features(df)

    feature_cols = [
        "temp_c",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month",
        "load_lag_1",
        "load_lag_24",
        "load_roll_mean_24",
    ]
    X = df[feature_cols].to_numpy()
    y = df["load"].to_numpy()

    models: dict[str, Pipeline] = {
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=seed,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                )
            ]
        ),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[ForecastResult] = []

    for model_name, pipe in models.items():
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_mape: list[float] = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            fold_mae.append(float(mean_absolute_error(y_test, pred)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, pred))))
            fold_mape.append(_safe_mape(y_test, pred))

        results.append(
            ForecastResult(
                model_name=model_name,
                mae=float(np.mean(fold_mae)),
                rmse=float(np.mean(fold_rmse)),
                mape=float(np.mean(fold_mape)),
            )
        )

    return sorted(results, key=lambda r: r.rmse)


def load_dataset(csv_path: Path | None) -> pd.DataFrame:
    """
    Expected CSV columns:
      - datetime: parseable timestamp
      - load: numeric
      - temp_c: numeric (optional; if missing will be created as NaN and dropped later)
    """
    if csv_path is None:
        return make_synthetic_hourly_dataset()

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns or "load" not in df.columns:
        raise ValueError("CSV must include at least columns: datetime, load")
    if "temp_c" not in df.columns:
        df["temp_c"] = np.nan
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Energy load forecasting baseline pipeline.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with datetime, load, temp_c.")
    parser.add_argument("--splits", type=int, default=5, help="Number of time-series CV splits.")
    args = parser.parse_args()

    df = load_dataset(Path(args.csv) if args.csv else None)
    results = train_and_evaluate(df, n_splits=args.splits)

    print("Model performance (time-series CV; lower is better):")
    for r in results:
        print(f"- {r.model_name}: MAE={r.mae:.2f}, RMSE={r.rmse:.2f}, MAPE={r.mape:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

