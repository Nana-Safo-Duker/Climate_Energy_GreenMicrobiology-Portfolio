from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "synthetic_renewable_timeseries.csv"


def generate_synthetic_dataset(n_days: int = 180, freq: str = "h", seed: int = 7) -> pd.DataFrame:
    """
    Create a realistic-ish renewable power time series with meteorological drivers.
    This keeps the repo runnable without external data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days * 24, freq=freq)
    n = len(idx)

    hour = idx.hour.values
    dayofyear = idx.dayofyear.values

    # Irradiance proxy: daily cycle + seasonal modulation + cloud noise
    seasonal = 0.65 + 0.35 * np.sin(2 * np.pi * (dayofyear / 365.25))
    diurnal = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)  # 0 at night, peak midday
    cloud = np.clip(rng.normal(0.0, 0.18, size=n), -0.6, 0.6)
    irradiance = np.clip(seasonal * diurnal * (1 + cloud), 0, None)

    # Wind speed proxy: seasonal + autocorrelated component + gust noise
    base_wind = 7 + 2.0 * np.sin(2 * np.pi * (dayofyear / 365.25 + 0.25))
    ar = rng.normal(0, 0.8, size=n)
    for i in range(1, n):
        ar[i] = 0.85 * ar[i - 1] + ar[i]
    wind_speed = np.clip(base_wind + ar + rng.normal(0, 1.0, size=n), 0, None)

    temperature = 12 + 10 * np.sin(2 * np.pi * (dayofyear / 365.25 - 0.1)) + rng.normal(0, 1.5, size=n)
    humidity = np.clip(55 + 20 * np.sin(2 * np.pi * (dayofyear / 365.25 + 0.05)) + rng.normal(0, 6, size=n), 10, 100)

    # Convert drivers into PV and wind power-like signals
    pv_power = np.clip(irradiance ** 1.15 + rng.normal(0, 0.03, size=n), 0, None)

    # Wind turbine curve-ish: cut-in ~3 m/s, rated around 12 m/s, cut-out ignored
    wind_norm = np.clip((wind_speed - 3) / (12 - 3), 0, 1)
    wind_power = np.clip(wind_norm ** 3 + rng.normal(0, 0.04, size=n), 0, 1)

    # Target: aggregated renewable output (normalized), add measurement noise
    renewable_power = np.clip(0.55 * pv_power + 0.45 * wind_power + rng.normal(0, 0.03, size=n), 0, 1.2)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "irradiance": irradiance,
            "wind_speed": wind_speed,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "hour": hour,
            "dayofyear": dayofyear,
            "renewable_power": renewable_power,
        }
    )

    # Inject a small number of missing values to make preprocessing realistic
    for col in ["irradiance", "wind_speed", "temperature_c", "humidity_pct"]:
        mask = rng.random(n) < 0.01
        df.loc[mask, col] = np.nan

    return df


def make_supervised(df: pd.DataFrame, horizon_hours: int = 1, n_lags: int = 24) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["target"] = df["renewable_power"].shift(-horizon_hours)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["renewable_power"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def rolling_backtest(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    baseline: Pipeline,
    optimized: GridSearchCV,
    test_size: int,
    n_splits: int,
) -> pd.DataFrame:
    """
    Rolling evaluation:
    - Walk forward in time with fixed-size test blocks
    - Refit both models each round
    - Store per-round metrics for comparison
    """
    n = len(data)
    fold_starts = np.linspace(0, n - test_size, num=n_splits + 1, dtype=int)[1:]
    rows = []

    for i, start in enumerate(fold_starts, start=1):
        train = data.iloc[:start]
        test = data.iloc[start : start + test_size]

        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        baseline.fit(X_train, y_train)
        yhat_base = baseline.predict(X_test)

        optimized.fit(X_train, y_train)
        yhat_opt = optimized.predict(X_test)

        rows.append(
            {
                "fold": i,
                "test_start": test["timestamp"].iloc[0],
                "test_end": test["timestamp"].iloc[-1],
                "baseline_mae": mean_absolute_error(y_test, yhat_base),
                "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, yhat_base))),
                "optimized_mae": mean_absolute_error(y_test, yhat_opt),
                "optimized_rmse": float(np.sqrt(mean_squared_error(y_test, yhat_opt))),
                "best_params": str(getattr(optimized, "best_params_", None)),
            }
        )

    return pd.DataFrame(rows)


def build_models(*, n_inner_splits: int = 5, fast: bool = False) -> tuple[Pipeline, GridSearchCV]:
    numeric_features = [
        "irradiance",
        "wind_speed",
        "temperature_c",
        "humidity_pct",
        "dayofyear",
        *[f"lag_{i}" for i in range(1, 25)],
    ]
    categorical_features = ["hour"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    baseline = Pipeline(steps=[("preprocess", preprocess), ("model", Ridge(alpha=1.0, random_state=0))])

    rf = RandomForestRegressor(
        random_state=0,
        n_jobs=-1,
    )
    optimized_pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", rf)])

    if fast:
        param_grid = {
            "model__n_estimators": [200],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 5],
        }
    else:
        param_grid = {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 3, 5],
        }
    inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)
    optimized = GridSearchCV(
        optimized_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
    )

    return baseline, optimized


def plot_results(metrics: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    long_mae = metrics.melt(
        id_vars=["fold", "test_start"],
        value_vars=["baseline_mae", "optimized_mae"],
        var_name="model",
        value_name="mae",
    )
    long_mae["model"] = long_mae["model"].map({"baseline_mae": "Baseline (Ridge)", "optimized_mae": "Optimized (RF GridSearch)"})

    plt.figure(figsize=(10, 4.8))
    sns.lineplot(data=long_mae, x="test_start", y="mae", hue="model", marker="o")
    plt.title("Rolling backtest MAE (lower is better)")
    plt.xlabel("Test window start")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(figures_dir / "rolling_backtest_mae.png", dpi=200)
    plt.close()

    summary = pd.DataFrame(
        {
            "model": ["Baseline (Ridge)", "Optimized (RF GridSearch)"],
            "MAE_mean": [metrics["baseline_mae"].mean(), metrics["optimized_mae"].mean()],
            "RMSE_mean": [metrics["baseline_rmse"].mean(), metrics["optimized_rmse"].mean()],
        }
    )
    summary.to_csv(figures_dir / "metrics_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible renewable forecasting backtest (synthetic by default).")
    parser.add_argument("--n-days", type=int, default=180, help="Days of hourly data to generate if dataset not found.")
    parser.add_argument("--horizon-hours", type=int, default=1, help="Forecast horizon in hours.")
    parser.add_argument("--test-size", type=int, default=7 * 24, help="Test block size in rows (default: 7 days).")
    parser.add_argument("--splits", type=int, default=6, help="Number of rolling test blocks.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller grid search + fewer CV splits (useful for quick demos).",
    )
    args = parser.parse_args()

    paths = Paths(root=Path(__file__).resolve().parents[1])
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    if paths.dataset_path.exists():
        df = pd.read_csv(paths.dataset_path, parse_dates=["timestamp"])
    else:
        df = generate_synthetic_dataset(n_days=args.n_days, seed=args.seed)
        df.to_csv(paths.dataset_path, index=False)

    sup = make_supervised(df, horizon_hours=args.horizon_hours, n_lags=24)
    feature_cols = [c for c in sup.columns if c not in {"target", "timestamp", "renewable_power"}]

    baseline, optimized = build_models(n_inner_splits=(3 if args.fast else 5), fast=args.fast)
    metrics = rolling_backtest(
        data=sup,
        feature_cols=feature_cols,
        target_col="target",
        baseline=baseline,
        optimized=optimized,
        test_size=args.test_size,
        n_splits=args.splits,
    )

    metrics.to_csv(paths.reports_dir / "rolling_backtest_metrics.csv", index=False)
    plot_results(metrics, paths.figures_dir)

    print("Saved:")
    print(f"- Dataset: {paths.dataset_path}")
    print(f"- Metrics: {paths.reports_dir / 'rolling_backtest_metrics.csv'}")
    print(f"- Figure:  {paths.figures_dir / 'rolling_backtest_mae.png'}")
    print(f"- Summary: {paths.figures_dir / 'metrics_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

