"""
ClimaX Research Review — Reproducible Analysis Script

Educational companion to Nguyen et al. (ICML 2023), "ClimaX: A foundation
model for weather and climate". This script does NOT download CMIP6/ERA5 or
run the official microsoft/ClimaX checkpoints. Instead it builds transparent
synthetic benchmarks that illustrate the paper's evaluation logic:

1) Global forecast skill vs lead time (foundation vs task-specific vs NWP-like)
2) Paired statistical tests of absolute errors
3) ClimateBench-style projection score comparison
4) Downscaling RMSE comparison
5) Simple scaling-law illustration (model size vs skill)

Outputs are written to data/processed/ and outputs/ for reporting.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass(frozen=True)
class Config:
    seed: int = 2023
    n_samples: int = 250
    lead_hours: tuple[int, ...] = (6, 12, 24, 48, 72, 120, 168, 240, 336, 720)
    variables: tuple[str, ...] = ("t2m", "t850", "z500", "u10")


def ensure_directories() -> Path:
    """Create expected project directories and return project root."""
    root = Path(__file__).resolve().parent
    for folder in [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "outputs",
        root / "assets",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        if folder.name in {"raw", "processed"} or folder.name == "outputs":
            (folder / ".gitkeep").touch(exist_ok=True)
    return root


def _model_scale(model: str, lead_h: int) -> float:
    """
    Relative error scale by model family.

    Inspired by paper narrative (not official scores):
    - task_specific_dl: strong at short range, degrades faster at long leads
    - climax_like: competitive short/medium range, stronger long-horizon transfer
    - nwp_like: strong short range, relatively weaker at very long leads
    """
    if model == "task_specific_dl":
        return 1.05 + 0.0018 * lead_h
    if model == "climax_like":
        return 1.00 + 0.0011 * lead_h
    if model == "nwp_like":
        return 0.98 + 0.0016 * lead_h
    raise ValueError(f"Unknown model: {model}")


def _variable_difficulty(variable: str) -> float:
    difficulties = {"t2m": 1.0, "t850": 1.08, "z500": 0.95, "u10": 1.12}
    return difficulties.get(variable, 1.0)


def build_forecast_dataset(cfg: Config) -> pd.DataFrame:
    """Synthetic absolute-error samples by lead time, variable, and model."""
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, float | int | str]] = []

    for variable in cfg.variables:
        for lead_h in cfg.lead_hours:
            for model in ("task_specific_dl", "climax_like", "nwp_like"):
                base = (0.85 + 0.008 * lead_h) * _variable_difficulty(variable)
                scale = _model_scale(model, lead_h)
                errors = scale * base * rng.lognormal(mean=0.0, sigma=0.32, size=cfg.n_samples)
                for i, err in enumerate(errors):
                    rows.append(
                        {
                            "sample_id": i,
                            "variable": variable,
                            "lead_hours": lead_h,
                            "model": model,
                            "abs_error": float(err),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_forecast_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std of absolute error by model, variable, and lead time."""
    summary = (
        df.groupby(["model", "variable", "lead_hours"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            std_ae=("abs_error", "std"),
            rmse=("abs_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        )
        .sort_values(["variable", "lead_hours", "model"])
    )
    return summary


def paired_ttests_climax_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-tests of absolute errors: climax_like vs task_specific_dl
    (alternative: climax errors are smaller -> baseline - climax > 0).
    """
    rows: list[dict[str, float | int | str]] = []
    pivot = df.pivot_table(
        index=["sample_id", "variable", "lead_hours"],
        columns="model",
        values="abs_error",
        aggfunc="first",
    ).reset_index()

    for (variable, lead_h), sub in pivot.groupby(["variable", "lead_hours"]):
        t_stat, p_value = stats.ttest_rel(
            sub["task_specific_dl"].to_numpy(),
            sub["climax_like"].to_numpy(),
            alternative="greater",
        )
        rows.append(
            {
                "variable": variable,
                "lead_hours": int(lead_h),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "mean_diff_baseline_minus_climax": float(
                    (sub["task_specific_dl"] - sub["climax_like"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["variable", "lead_hours"])


def build_climatebench_scores(cfg: Config) -> pd.DataFrame:
    """
    Synthetic ClimateBench-style total scores (higher is better).
    Illustrates transfer of ClimaX-like attention to unseen projection variables.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    models = ["MLP", "CNN", "RandomForest", "ClimaX_like"]
    targets = ["tas", "pr", "diurnal_temperature_range"]
    rows = []
    base = {
        "MLP": 0.62,
        "CNN": 0.71,
        "RandomForest": 0.68,
        "ClimaX_like": 0.78,
    }
    for target in targets:
        for model in models:
            score = np.clip(rng.normal(base[model], 0.03), 0.0, 1.0)
            rows.append({"target": target, "model": model, "score": float(score)})
    return pd.DataFrame(rows)


def build_downscaling_metrics(cfg: Config) -> pd.DataFrame:
    """Synthetic downscaling RMSE (lower is better) for key variables."""
    rng = np.random.default_rng(cfg.seed + 11)
    models = ["Bilinear", "CNN_baseline", "ClimaX_like"]
    variables = ["t2m", "t850", "z500"]
    # Relative RMSE levels inspired by "ClimaX compares favorably" narrative
    relative = {"Bilinear": 1.25, "CNN_baseline": 1.05, "ClimaX_like": 0.92}
    rows = []
    for variable in variables:
        for model in models:
            rmse = relative[model] * _variable_difficulty(variable) * rng.normal(1.0, 0.04)
            rows.append(
                {
                    "variable": variable,
                    "model": model,
                    "rmse": float(max(rmse, 0.05)),
                }
            )
    return pd.DataFrame(rows)


def build_scaling_curve(cfg: Config) -> pd.DataFrame:
    """Toy scaling law: larger pretrained models -> lower 3-day forecast MAE."""
    rng = np.random.default_rng(cfg.seed + 13)
    params_m = np.array([5, 15, 50, 100, 200], dtype=float)
    # Power-law-ish improvement with diminishing returns
    mae = 2.4 * (params_m**-0.18) + rng.normal(0.0, 0.03, size=len(params_m))
    return pd.DataFrame(
        {
            "params_millions": params_m,
            "mae_3day_t850": np.clip(mae, 0.5, None),
            "pretrain_datasets": [1, 2, 3, 4, 5],
        }
    )


def plot_forecast_skill(summary: pd.DataFrame, out_png: Path, assets_png: Path) -> None:
    """Lead-time MAE curves for t2m across models."""
    sub = summary[summary["variable"] == "t2m"].copy()
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=sub,
        x="lead_hours",
        y="mae",
        hue="model",
        marker="o",
        linewidth=2,
    )
    plt.title("Synthetic global forecast skill (T2m) — lower MAE is better")
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Mean absolute error (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    assets_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(assets_png, dpi=150)
    plt.close()


def plot_climatebench(scores: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=scores, x="target", y="score", hue="model")
    plt.title("Synthetic ClimateBench-style projection scores (higher is better)")
    plt.xlabel("Target variable")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_downscaling(metrics: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics, x="variable", y="rmse", hue="model")
    plt.title("Synthetic downscaling RMSE (lower is better)")
    plt.xlabel("Variable")
    plt.ylabel("RMSE (a.u.)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_scaling(scaling: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(scaling["params_millions"], scaling["mae_3day_t850"], marker="o", linewidth=2)
    plt.xscale("log")
    plt.title("Synthetic scaling: model size vs 3-day T850 MAE")
    plt.xlabel("Parameters (millions, log scale)")
    plt.ylabel("MAE (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ClimaX-inspired synthetic evaluation for research review."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <project>/outputs)",
    )
    args = parser.parse_args()

    root = ensure_directories()
    outdir = Path(args.outdir) if args.outdir else root / "outputs"
    processed = root / "data" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    forecast_df = build_forecast_dataset(cfg)
    summary = summarize_forecast_skill(forecast_df)
    ttests = paired_ttests_climax_vs_baseline(forecast_df)
    climatebench = build_climatebench_scores(cfg)
    downscaling = build_downscaling_metrics(cfg)
    scaling = build_scaling_curve(cfg)

    # Persist tabular outputs
    forecast_df.to_csv(processed / "forecast_abs_errors.csv", index=False)
    summary.to_csv(processed / "forecast_skill_summary.csv", index=False)
    ttests.to_csv(processed / "forecast_paired_ttests.csv", index=False)
    climatebench.to_csv(processed / "climatebench_scores.csv", index=False)
    downscaling.to_csv(processed / "downscaling_rmse.csv", index=False)
    scaling.to_csv(processed / "scaling_curve.csv", index=False)

    summary.to_csv(outdir / "forecast_skill_summary.csv", index=False)
    ttests.to_csv(outdir / "forecast_paired_ttests.csv", index=False)
    climatebench.to_csv(outdir / "climatebench_scores.csv", index=False)
    downscaling.to_csv(outdir / "downscaling_rmse.csv", index=False)
    scaling.to_csv(outdir / "scaling_curve.csv", index=False)

    assets = root / "assets"
    plot_forecast_skill(
        summary,
        outdir / "forecast_skill_t2m.png",
        assets / "overview.png",
    )
    plot_climatebench(climatebench, outdir / "climatebench_scores.png")
    plot_downscaling(downscaling, outdir / "downscaling_rmse.png")
    plot_scaling(scaling, outdir / "scaling_curve.png")

    # Mirror key figures into assets/ for README rendering
    for name in (
        "forecast_skill_t2m.png",
        "climatebench_scores.png",
        "downscaling_rmse.png",
        "scaling_curve.png",
    ):
        shutil.copy2(outdir / name, assets / name)

    # Concise console report
    t2m = summary[summary["variable"] == "t2m"]
    long_lead = t2m[t2m["lead_hours"] == max(cfg.lead_hours)].sort_values("mae")
    print("Wrote outputs to:", outdir.resolve())
    print("Wrote processed tables to:", processed.resolve())
    print("\nLongest-lead T2m MAE ranking (synthetic):")
    print(long_lead[["model", "mae", "rmse"]].to_string(index=False))
    sig = ttests[(ttests["variable"] == "t2m") & (ttests["p_value"] < 0.05)]
    print(f"\nT2m leads with climax_like < task_specific_dl (p<0.05): {len(sig)}/{len(cfg.lead_hours)}")
    print("\nNote: synthetic demo for education/review — not official ClimaX scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
