from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Config:
    seed: int = 101
    n_samples: int = 200
    lead_hours: tuple[int, ...] = (6, 12, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240)


def _make_synthetic_errors(
    rng: np.random.Generator, n: int, lead_h: int, model: str
) -> np.ndarray:
    """
    Create synthetic absolute errors that increase with lead time.

    This is a toy stand-in for a "truth vs forecast" evaluation pipeline when
    you don't want to bundle large reanalysis datasets in a repository.
    """
    base = 1.0 + 0.010 * lead_h
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)

    if model == "baseline":
        scale = 1.15
    elif model == "graphcast_like":
        scale = 0.95
    else:
        raise ValueError(f"Unknown model: {model}")

    return scale * base * noise


def build_dataset(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict[str, float | int | str]] = []
    for lead_h in cfg.lead_hours:
        e_base = _make_synthetic_errors(rng, cfg.n_samples, lead_h, "baseline")
        e_gc = _make_synthetic_errors(rng, cfg.n_samples, lead_h, "graphcast_like")

        for i in range(cfg.n_samples):
            rows.append(
                {
                    "sample_id": i,
                    "lead_hours": lead_h,
                    "baseline_abs_error": float(e_base[i]),
                    "graphcast_like_abs_error": float(e_gc[i]),
                }
            )

    df = pd.DataFrame(rows)
    df["error_diff"] = df["baseline_abs_error"] - df["graphcast_like_abs_error"]
    return df


def summarize_by_lead(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("lead_hours", as_index=False)
    out = g.agg(
        baseline_mae=("baseline_abs_error", "mean"),
        graphcast_like_mae=("graphcast_like_abs_error", "mean"),
        diff_mean=("error_diff", "mean"),
        diff_std=("error_diff", "std"),
    )
    return out.sort_values("lead_hours")


def paired_ttest_by_lead(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for lead_h, sub in df.groupby("lead_hours"):
        t, p = stats.ttest_rel(
            sub["baseline_abs_error"].to_numpy(),
            sub["graphcast_like_abs_error"].to_numpy(),
            alternative="greater",
        )
        rows.append({"lead_hours": int(lead_h), "t_stat": float(t), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("lead_hours")


def plot_skill(summary: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = summary["lead_hours"].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(x, summary["baseline_mae"], marker="o", label="Baseline (MAE)")
    plt.plot(x, summary["graphcast_like_mae"], marker="o", label="GraphCast-like (MAE)")
    plt.title("Toy skill curve (lower MAE is better)")
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Mean absolute error (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    from pathlib import Path as _P
    _assets = _P(__file__).resolve().parents[1] / "assets"
    _assets.mkdir(parents=True, exist_ok=True)
    plt.savefig(_assets / "overview.png", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Toy GraphCast-style skill demo.")
    parser.add_argument("--outdir", default="outputs", help="Directory for generated files.")
    args = parser.parse_args()

    cfg = Config()
    df = build_dataset(cfg)
    summary = summarize_by_lead(df)
    ttests = paired_ttest_by_lead(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "toy_skill_samples.csv", index=False)
    summary.to_csv(outdir / "toy_skill_summary.csv", index=False)
    ttests.to_csv(outdir / "toy_skill_ttests.csv", index=False)
    plot_skill(summary, outdir / "toy_skill_curve.png")

    best = summary.iloc[0].to_dict()
    worst = summary.iloc[-1].to_dict()
    print("Wrote outputs to:", outdir.resolve())
    print("Best lead (shortest):", best)
    print("Worst lead (longest):", worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

