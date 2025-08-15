"""
Continuous Ensemble Weather Forecasting — educational demo pipeline.

This script does **not** retrain the full diffusion U-Net from Andrae et al. (2024).
Instead, it implements a lightweight, reproducible analogue of the paper's ideas:

1. Lead-time–conditioned forecasting of a spatiotemporal weather proxy field.
2. Continuous ensemble trajectories via fixed / autocorrelated noise seeds.
3. Autoregressive rollouts with continuous interpolation (ARCI-style hybrid).
4. Probabilistic metrics: RMSE (ensemble mean), CRPS (Gaussian approx), SSR.

Replace the synthetic generator with ERA5 / WeatherBench loaders for research use.
Paper: https://doi.org/10.48550/arXiv.2410.05431
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Config:
    n_lat: int = 16
    n_lon: int = 32
    n_hours: int = 240  # 10 days at hourly resolution
    n_ens: int = 30
    ar_step_hours: int = 24
    interp_hours: int = 6
    rho: float = float(np.log(10.0))  # Ornstein–Uhlenbeck correlation (paper-style)
    noise_scale: float = 2.5
    random_state: int = 42
    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lat_lon_grid(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-75, 75, n_lat)
    lon = np.linspace(-180, 180, n_lon, endpoint=False)
    lon_g, lat_g = np.meshgrid(lon, lat)
    return lat_g, lon_g


def true_field(
    t_hours: float,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthetic t850-like proxy (°C): seasonal wave + travelling pattern + local noise.
    """
    seasonal = 12.0 * np.sin(2 * np.pi * (t_hours % 8760) / 8760.0)
    wave = 6.0 * np.sin(2 * np.pi * (lon_g / 360.0) - 2 * np.pi * t_hours / 120.0)
    meridional = 8.0 * np.cos(np.deg2rad(lat_g))
    field = 273.15 - 20.0 + seasonal + wave + meridional
    if rng is not None:
        field = field + rng.normal(0.0, 0.4, size=field.shape)
    return field.astype(np.float64)


def score_conditioned_mean(
    init0: np.ndarray,
    init_m: np.ndarray,
    lead_h: float,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
) -> np.ndarray:
    """
    Cheap stand-in for a lead-time–conditioned denoiser mean prediction.

    Blends a persistence/tendency forecast with the known climatological attractor
    of the synthetic truth (stronger climatology weight as lead time grows).
    """
    tendency = init0 - init_m
    persistence = init0 + tendency * (lead_h / 6.0) * 0.15
    climate = true_field(lead_h, lat_g, lon_g, rng=None)
    # Correlation decays with lead time → more weight on climate (mirrors CI weakness)
    w = np.exp(-lead_h / 72.0)
    return w * persistence + (1.0 - w) * climate


def ou_noise_process(
    n_times: int,
    shape: tuple[int, ...],
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample autocorrelated Gaussian noise Z(t) for Continuous Ensemble Forecasting."""
    z = np.zeros((n_times, *shape), dtype=np.float64)
    z[0] = rng.normal(0.0, 1.0, size=shape)
    for i in range(1, n_times):
        dt = 1.0
        alpha = np.exp(-rho * dt)
        innov = rng.normal(0.0, 1.0, size=shape)
        z[i] = alpha * z[i - 1] + np.sqrt(max(1.0 - alpha**2, 0.0)) * innov
    return z


def continuous_ensemble_forecast(
    init0: np.ndarray,
    init_m: np.ndarray,
    lead_hours: np.ndarray,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    n_ens: int,
    rho: float,
    noise_scale: float,
    rng: np.random.Generator,
    fixed_noise: bool = False,
) -> np.ndarray:
    """
    Algorithm 1 / 2 analogue: parallel lead-time forecasts with shared or OU noise.

    Returns array shaped (n_ens, n_leads, n_lat, n_lon).
    """
    n_leads = len(lead_hours)
    shape = init0.shape
    if fixed_noise or rho <= 0:
        z0 = rng.normal(0.0, 1.0, size=(n_ens, *shape))
        noise = np.repeat(z0[:, None, ...], n_leads, axis=1)
    else:
        noise = np.stack(
            [ou_noise_process(n_leads, shape, rho, rng) for _ in range(n_ens)],
            axis=0,
        )

    out = np.zeros((n_ens, n_leads, *shape), dtype=np.float64)
    for j, lead in enumerate(lead_hours):
        mean = score_conditioned_mean(init0, init_m, float(lead), lat_g, lon_g)
        # lead-dependent residual scale (underdispersion tendency like diffusion SSR < 1)
        scale = noise_scale * (0.7 + 0.3 * (1.0 - np.exp(-float(lead) / 48.0)))
        out[:, j] = mean[None, ...] + scale * noise[:, j]
    return out


def arci_forecast(
    init0: np.ndarray,
    init_m: np.ndarray,
    total_hours: int,
    ar_step: int,
    interp: int,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    n_ens: int,
    rho: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 3 analogue: autoregressive anchors every `ar_step` hours with
    continuous interpolation at `interp`-hour spacing inside each block.
    """
    leads_block = np.arange(interp, ar_step + 1, interp, dtype=float)
    n_blocks = int(np.ceil(total_hours / ar_step))
    all_members: list[np.ndarray] = []
    all_leads: list[float] = []

    cur0, cur_m = init0.copy(), init_m.copy()
    t0 = 0.0
    for _ in range(n_blocks):
        block = continuous_ensemble_forecast(
            cur0,
            cur_m,
            leads_block,
            lat_g,
            lon_g,
            n_ens,
            rho,
            noise_scale,
            rng,
            fixed_noise=False,
        )
        for j, lead in enumerate(leads_block):
            all_members.append(block[:, j])
            all_leads.append(t0 + float(lead))
        # Advance AR anchor using ensemble mean at ar_step
        cur_m = cur0
        cur0 = block[:, -1].mean(axis=0)
        t0 += ar_step

    stacked = np.stack(all_members, axis=1)  # (ens, time, lat, lon)
    return stacked, np.asarray(all_leads, dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def crps_gaussian(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for univariate Gaussian forecasts (Gneiting & Raftery)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (obs - mu) / sigma
    scores = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(scores))


def spread_skill_ratio(ens: np.ndarray, truth: np.ndarray) -> float:
    """
    SSR = ensemble spread / RMSE(ensemble mean).
    Well-calibrated ensembles target SSR ≈ 1.
    """
    ens_mean = ens.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.var(ens, axis=0, ddof=1))))
    skill = rmse(ens_mean, truth)
    return float(spread / max(skill, 1e-8))


def evaluate_against_truth(
    ens: np.ndarray,
    leads: np.ndarray,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for j, lead in enumerate(leads):
        truth = true_field(float(lead), lat_g, lon_g, rng=rng)
        members = ens[:, j]
        mu = members.mean(axis=0)
        sigma = members.std(axis=0, ddof=1)
        rows.append(
            {
                "lead_hours": float(lead),
                "rmse": rmse(mu, truth),
                "crps": crps_gaussian(truth, mu, sigma),
                "ssr": spread_skill_ratio(members, truth),
                "mean_temporal_ready": True,
            }
        )
    return pd.DataFrame(rows)


def temporal_difference(series: np.ndarray) -> float:
    """Mean absolute hour-to-hour change (paper continuity diagnostic)."""
    if series.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(series, axis=0))))


def plot_overview(
    metrics: pd.DataFrame,
    ens: np.ndarray,
    leads: np.ndarray,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    rng: np.random.Generator,
    out_path: Path,
) -> None:
    truth_last = true_field(float(leads[-1]), lat_g, lon_g, rng=rng)
    mean_last = ens[:, -1].mean(axis=0)
    member0 = ens[0, -1]

    # Point time series at mid-domain for continuity view
    i, j = ens.shape[2] // 2, ens.shape[3] // 2
    ens_ts = ens[:, :, i, j]
    truth_ts = np.array(
        [true_field(float(h), lat_g, lon_g, rng=None)[i, j] for h in leads]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(metrics["lead_hours"], metrics["rmse"], label="RMSE", color="#1f4e79")
    ax.plot(metrics["lead_hours"], metrics["crps"], label="CRPS", color="#c45c26")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Score")
    ax.set_title("Probabilistic skill vs lead time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(metrics["lead_hours"], metrics["ssr"], color="#2a6f4e")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("SSR")
    ax.set_title("Spread/Skill Ratio (target ≈ 1)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for m in range(min(8, ens_ts.shape[0])):
        ax.plot(leads, ens_ts[m], color="#7aa2c4", alpha=0.45, linewidth=1)
    ax.plot(leads, ens_ts.mean(axis=0), color="#1f4e79", linewidth=2, label="Ens. mean")
    ax.plot(leads, truth_ts, color="#c45c26", linewidth=2, label="Truth proxy")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("t850 proxy (K)")
    ax.set_title("Continuous ensemble trajectories (grid center)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    vmax = max(float(np.max(np.abs(truth_last - mean_last))), 1.0)
    im = ax.imshow(member0 - truth_last, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Member − truth at final lead")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Continuous Ensemble Weather Forecasting — Educational Demo",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_skill_curves(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(metrics["lead_hours"], metrics["rmse"], color="#1f4e79", lw=2, label="RMSE")
    ax.plot(metrics["lead_hours"], metrics["crps"], color="#c45c26", lw=2, label="CRPS")
    ax.fill_between(
        metrics["lead_hours"],
        metrics["crps"],
        metrics["rmse"],
        color="#1f4e79",
        alpha=0.08,
    )
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Ensemble-mean RMSE and CRPS across lead times")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_noise_comparison(
    fixed: np.ndarray,
    ou: np.ndarray,
    short_leads: np.ndarray,
    truth_ts: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(short_leads, fixed[0], color="#1f4e79", lw=2, label="Fixed noise (Alg. 1)")
    ax.plot(short_leads, ou[0], color="#2a6f4e", lw=2, label="OU noise (Alg. 2)")
    ax.plot(short_leads, truth_ts, color="#c45c26", ls="--", lw=2, label="Truth proxy")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("t850 proxy (K)")
    ax.set_title("Temporal continuity under different driving-noise processes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_field_gallery(
    ens: np.ndarray,
    leads: np.ndarray,
    lat_g: np.ndarray,
    lon_g: np.ndarray,
    out_path: Path,
) -> None:
    pick_hours = [6, 24, 72, 168]
    idxs = [int(np.argmin(np.abs(leads - h))) for h in pick_hours]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for col, (h, idx) in enumerate(zip(pick_hours, idxs)):
        truth = true_field(float(leads[idx]), lat_g, lon_g, rng=None)
        mean = ens[:, idx].mean(axis=0)
        vmin = min(float(truth.min()), float(mean.min()))
        vmax = max(float(truth.max()), float(mean.max()))
        axes[0, col].imshow(truth, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"Truth @ {int(leads[idx])} h")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        im = axes[1, col].imshow(mean, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"Ens. mean @ {int(leads[idx])} h")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    axes[0, 0].set_ylabel("Truth")
    axes[1, 0].set_ylabel("Forecast")
    fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="K")
    fig.suptitle("Lead-time gallery: truth vs continuous-ensemble mean", y=0.98)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_spread(
    ens: np.ndarray,
    leads: np.ndarray,
    out_path: Path,
) -> None:
    i, j = ens.shape[2] // 2, ens.shape[3] // 2
    ens_ts = ens[:, :, i, j]
    q10 = np.percentile(ens_ts, 10, axis=0)
    q50 = np.percentile(ens_ts, 50, axis=0)
    q90 = np.percentile(ens_ts, 90, axis=0)
    truth_ts = np.array(
        [
            true_field(
                float(h),
                *lat_lon_grid(ens.shape[2], ens.shape[3]),
                rng=None,
            )[i, j]
            for h in leads
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(leads, q10, q90, color="#7aa2c4", alpha=0.35, label="10–90% ensemble")
    ax.plot(leads, q50, color="#1f4e79", lw=2, label="Ensemble median")
    ax.plot(leads, truth_ts, color="#c45c26", lw=2, label="Truth proxy")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("t850 proxy (K)")
    ax.set_title("Ensemble uncertainty band at domain center")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_method_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.4, 1.1, "Initial\nconditions\nX(Ω)"),
        (2.6, 1.1, "Lead-time\nconditioned\nscore / mean"),
        (4.8, 1.1, "Correlated\nnoise\nZ(t)"),
        (7.0, 1.1, "Parallel\nODE / map\nsampling"),
        (8.9, 1.1, "Ensemble\ntrajectories\nX(T)"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                1.5,
                1.2,
                fill=True,
                facecolor="#e8eef5",
                edgecolor="#1f4e79",
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(x + 0.75, y + 0.6, text, ha="center", va="center", fontsize=8, zorder=3)

    for x0 in [1.9, 4.1, 6.3, 8.5]:
        ax.annotate(
            "",
            xy=(x0 + 0.55, 1.7),
            xytext=(x0, 1.7),
            arrowprops=dict(arrowstyle="->", color="#c45c26", lw=1.6),
        )

    ax.text(
        5.0,
        0.35,
        "Continuous Ensemble Forecasting → optional ARCI (24 h anchors + fine fills)",
        ha="center",
        fontsize=9,
        color="#2a6f4e",
    )
    ax.set_title("Method sketch (educational)", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Config) -> dict:
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.assets_dir)
    rng = np.random.default_rng(cfg.random_state)
    lat_g, lon_g = lat_lon_grid(cfg.n_lat, cfg.n_lon)

    init0 = true_field(0.0, lat_g, lon_g, rng=rng)
    init_m = true_field(-6.0, lat_g, lon_g, rng=rng)

    ens, leads = arci_forecast(
        init0=init0,
        init_m=init_m,
        total_hours=cfg.n_hours,
        ar_step=cfg.ar_step_hours,
        interp=cfg.interp_hours,
        lat_g=lat_g,
        lon_g=lon_g,
        n_ens=cfg.n_ens,
        rho=cfg.rho,
        noise_scale=cfg.noise_scale,
        rng=rng,
    )

    metrics = evaluate_against_truth(ens, leads, lat_g, lon_g, rng)
    metrics_path = cfg.outputs_dir / "ensemble_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Continuity diagnostic: compare fixed vs OU noise on short CI horizon
    short_leads = np.arange(1, 25, dtype=float)
    fixed = continuous_ensemble_forecast(
        init0, init_m, short_leads, lat_g, lon_g, cfg.n_ens, 0.0, cfg.noise_scale, rng, True
    )
    ou = continuous_ensemble_forecast(
        init0, init_m, short_leads, lat_g, lon_g, cfg.n_ens, cfg.rho, cfg.noise_scale, rng, False
    )
    i, j = cfg.n_lat // 2, cfg.n_lon // 2
    continuity = {
        "temporal_diff_fixed_noise": temporal_difference(fixed[0, :, i, j]),
        "temporal_diff_ou_noise": temporal_difference(ou[0, :, i, j]),
        "temporal_diff_truth": temporal_difference(
            np.array([true_field(h, lat_g, lon_g, None)[i, j] for h in short_leads])
        ),
    }

    overview = cfg.assets_dir / "overview.png"
    plot_overview(metrics, ens, leads, lat_g, lon_g, rng, overview)
    # also mirror into outputs/
    plot_overview(metrics, ens, leads, lat_g, lon_g, rng, cfg.outputs_dir / "overview.png")

    # Extra README / gallery figures
    plot_skill_curves(metrics, cfg.assets_dir / "skill_curves.png")
    plot_skill_curves(metrics, cfg.outputs_dir / "skill_curves.png")
    plot_noise_comparison(
        fixed[:, :, i, j],
        ou[:, :, i, j],
        short_leads,
        np.array([true_field(h, lat_g, lon_g, None)[i, j] for h in short_leads]),
        cfg.assets_dir / "noise_comparison.png",
    )
    plot_noise_comparison(
        fixed[:, :, i, j],
        ou[:, :, i, j],
        short_leads,
        np.array([true_field(h, lat_g, lon_g, None)[i, j] for h in short_leads]),
        cfg.outputs_dir / "noise_comparison.png",
    )
    plot_field_gallery(ens, leads, lat_g, lon_g, cfg.assets_dir / "field_gallery.png")
    plot_field_gallery(ens, leads, lat_g, lon_g, cfg.outputs_dir / "field_gallery.png")
    plot_ensemble_spread(ens, leads, cfg.assets_dir / "ensemble_spread.png")
    plot_ensemble_spread(ens, leads, cfg.outputs_dir / "ensemble_spread.png")
    plot_method_schematic(cfg.assets_dir / "method_schematic.png")
    plot_method_schematic(cfg.outputs_dir / "method_schematic.png")

    summary = {
        "config": {**asdict(cfg), "outputs_dir": str(cfg.outputs_dir), "assets_dir": str(cfg.assets_dir)},
        "n_forecast_times": int(len(leads)),
        "mean_rmse": float(metrics["rmse"].mean()),
        "mean_crps": float(metrics["crps"].mean()),
        "mean_ssr": float(metrics["ssr"].mean()),
        "continuity": continuity,
        "metrics_csv": str(metrics_path),
        "overview_png": str(overview),
    }
    with open(cfg.outputs_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Educational continuous ensemble weather forecasting demo."
    )
    p.add_argument("--n-ens", type=int, default=Config.n_ens)
    p.add_argument("--n-hours", type=int, default=Config.n_hours)
    p.add_argument("--ar-step-hours", type=int, default=Config.ar_step_hours)
    p.add_argument("--interp-hours", type=int, default=Config.interp_hours)
    p.add_argument("--rho", type=float, default=Config.rho)
    p.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    p.add_argument("--random-state", type=int, default=Config.random_state)
    p.add_argument("--outputs-dir", type=str, default=str(Config.outputs_dir))
    p.add_argument("--assets-dir", type=str, default=str(Config.assets_dir))
    args = p.parse_args()
    return Config(
        n_ens=args.n_ens,
        n_hours=args.n_hours,
        ar_step_hours=args.ar_step_hours,
        interp_hours=args.interp_hours,
        rho=args.rho,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        outputs_dir=Path(args.outputs_dir),
        assets_dir=Path(args.assets_dir),
    )


def main() -> int:
    cfg = parse_args()
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
