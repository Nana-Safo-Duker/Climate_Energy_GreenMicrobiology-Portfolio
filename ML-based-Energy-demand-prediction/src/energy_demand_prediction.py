from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


UCI_APPLIANCES_URL = (
    "https://archive.ics.uci.edu/static/public/374/energy+efficiency.zip"
)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    random_state: int = 808


def load_or_make_demo_data(n_hours: int = 24 * 365, seed: int = 808) -> pd.DataFrame:
    """
    Create a realistic synthetic hourly demand dataset.

    Why synthetic? It guarantees this repo runs without credentials or private data.
    Replace with your utility/ISO dataset for a true replication.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame(index=idx)

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    temp_daily = 8 * np.sin(2 * np.pi * (df["hour"] / 24.0) - 1.2)
    temp_season = 12 * np.sin(2 * np.pi * (df.index.dayofyear / 365.25) - 0.5)
    df["temperature_c"] = (
        18
        + temp_daily
        + temp_season
        + rng.normal(0, 1.0, size=len(df))
    )

    is_weekend = (df["dow"] >= 5).astype(int)
    base = 1200 + 80 * np.cos(2 * np.pi * (df["hour"] / 24.0))
    heating = np.clip(18 - df["temperature_c"], 0, None) * 45
    cooling = np.clip(df["temperature_c"] - 22, 0, None) * 55
    weekend_drop = is_weekend * 120
    noise = rng.normal(0, 35, size=len(df))

    df["demand_mw"] = base + heating + cooling - weekend_drop + noise

    # Lags and rolling features (computed without leakage using shift)
    df["lag_1"] = df["demand_mw"].shift(1)
    df["lag_24"] = df["demand_mw"].shift(24)
    df["lag_168"] = df["demand_mw"].shift(168)
    df["roll_mean_24"] = df["demand_mw"].shift(1).rolling(24).mean()
    df["roll_std_24"] = df["demand_mw"].shift(1).rolling(24).std()

    # Drop early rows that don't have enough history
    df = df.dropna().reset_index(names="timestamp")
    return df


def time_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test


def build_model(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "ridge":
        reg = Ridge(alpha=1.0, random_state=808)
    elif model_name == "hgb":
        reg = HistGradientBoostingRegressor(
            random_state=808,
            max_depth=6,
            learning_rate=0.08,
        )
    else:
        raise ValueError(
            f"Unknown model_name={model_name!r} (expected 'ridge' or 'hgb')"
        )

    return Pipeline([("pre", pre), ("reg", reg)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Energy demand prediction demo (time-series safe features)."
    )
    parser.add_argument("--model", choices=["ridge", "hgb"], default="hgb")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    df = load_or_make_demo_data()

    target = "demand_mw"
    categorical_cols = ["dow", "month"]
    numeric_cols = [
        "hour",
        "temperature_c",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_std_24",
    ]

    train, test = time_split(df, train_frac=args.train_frac)
    X_train = train[numeric_cols + categorical_cols]
    y_train = train[target].to_numpy()
    X_test = test[numeric_cols + categorical_cols]
    y_test = test[target].to_numpy()

    model = build_model(
        args.model,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = evaluate(y_test, pred)
    print(f"Model: {args.model}")
    print(f"Rows: train={len(train):,} test={len(test):,}")
    print(f"MAE:  {metrics['mae']:.2f} MW")
    print(f"RMSE: {metrics['rmse']:.2f} MW")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
