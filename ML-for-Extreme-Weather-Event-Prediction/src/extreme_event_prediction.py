from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class Config:
    n_samples: int = 5000
    extreme_quantile: float = 0.95
    test_size: float = 0.25
    random_state: int = 909
    outputs_dir: Path = Path("outputs")


def make_synthetic_climate_table(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create a toy daily dataset with seasonal structure and noise.

    This is a stand-in for real climate covariates; replace with your own
    dataset.
    """
    t = np.arange(n)
    day_of_year = t % 365

    seasonal = np.sin(2 * np.pi * day_of_year / 365.0)
    temp = 15 + 10 * seasonal + rng.normal(0, 2.0, size=n)
    humidity = 60 - 15 * seasonal + rng.normal(0, 5.0, size=n)
    pressure = 1013 + rng.normal(0, 8.0, size=n)
    wind = np.clip(rng.gamma(shape=2.0, scale=1.5, size=n), 0, None)

    # Target (continuous): e.g., precipitation proxy with rare spikes
    base = 1.0 + 0.3 * (humidity / 100.0) + 0.1 * wind
    spikes = rng.binomial(1, 0.06, size=n) * rng.lognormal(
        mean=1.8, sigma=0.6, size=n
    )
    target = np.clip(base + spikes + rng.normal(0, 0.2, size=n), 0, None)

    return pd.DataFrame(
        {
            "t": t,
            "day_of_year": day_of_year,
            "temp_c": temp,
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "wind_ms": wind,
            "target": target,
        }
    )


def add_extreme_label(
    df: pd.DataFrame, target_col: str, q: float
) -> tuple[pd.DataFrame, float]:
    threshold = float(df[target_col].quantile(q))
    out = df.copy()
    out["is_extreme"] = (out[target_col] >= threshold).astype(int)
    return out, threshold


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, feature_cols),
        ],
        remainder="drop",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nPR-AUC (Average Precision) = {ap:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str
) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extreme event prediction baseline workflow."
    )
    parser.add_argument("--n-samples", type=int, default=Config.n_samples)
    parser.add_argument(
        "--extreme-quantile", type=float, default=Config.extreme_quantile
    )
    parser.add_argument("--test-size", type=float, default=Config.test_size)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--outputs-dir", type=str, default=str(Config.outputs_dir))
    args = parser.parse_args()

    cfg = Config(
        n_samples=args.n_samples,
        extreme_quantile=args.extreme_quantile,
        test_size=args.test_size,
        random_state=args.random_state,
        outputs_dir=Path(args.outputs_dir),
    )

    sns.set_theme(style="whitegrid")
    rng = np.random.default_rng(cfg.random_state)

    df = make_synthetic_climate_table(cfg.n_samples, rng)
    df, threshold = add_extreme_label(
        df, target_col="target", q=cfg.extreme_quantile
    )

    feature_cols = [
        "day_of_year",
        "temp_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_ms",
    ]
    X = df[feature_cols]
    y = df["is_extreme"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pre = build_preprocessor(feature_cols)

    models: dict[str, Pipeline] = {
        "logreg": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=cfg.random_state,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    ensure_dir(cfg.outputs_dir)

    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        roc = roc_auc_score(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)

        report = classification_report(
            y_test, pred, output_dict=True, zero_division=0
        )
        rows.append(
            {
                "model": name,
                "extreme_quantile": cfg.extreme_quantile,
                "threshold_target_value": threshold,
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "brier": brier,
                "precision_pos": report["1"]["precision"],
                "recall_pos": report["1"]["recall"],
                "f1_pos": report["1"]["f1-score"],
                "support_pos": report["1"]["support"],
            }
        )

        plot_pr_curve(
            y_test,
            prob,
            cfg.outputs_dir / f"pr_curve_{name}.png",
            title=f"Precision–Recall curve ({name})",
        )
        plot_calibration(
            y_test,
            prob,
            cfg.outputs_dir / f"calibration_{name}.png",
            title=f"Calibration curve ({name})",
        )

    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(cfg.outputs_dir / "metrics.csv", index=False)

    # Lightweight EDA figure
    plt.figure(figsize=(7, 4))
    sns.histplot(df["target"], bins=50, kde=False)
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{cfg.extreme_quantile:.0%} threshold",
    )
    plt.title("Synthetic target distribution (with extreme threshold)")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "target_distribution.png", dpi=160)
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    plt.savefig(assets / "overview.png", dpi=150)
    plt.close()

    print("Saved metrics to:", cfg.outputs_dir / "metrics.csv")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
