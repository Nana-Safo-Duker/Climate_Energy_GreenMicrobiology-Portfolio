"""
DeepSD-inspired stacked SRCNN pipeline for precipitation downscaling.

Educational reimplementation of ideas from:
  Vandal et al. (2017). DeepSD: Generating High Resolution Climate Change
  Projections through Single Image Super-Resolution. ACM SIGKDD.

Default mode uses synthetic precipitation + elevation fields so the pipeline
runs without proprietary PRISM / GTOPO30 downloads. Swap in real gridded
arrays when available (see README).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 2017
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    train_days: int = 240
    val_days: int = 60
    test_days: int = 60
    # Final high-resolution grid (toy CONUS-like tile)
    high_h: int = 64
    high_w: int = 128
    # Each SRCNN stage does 2x; three stages => 8x overall (1.0° -> 1/8°)
    stages: int = 3
    patch_size: int = 32
    patches_per_day: int = 8
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")

    @property
    def low_h(self) -> int:
        return self.high_h // (2 ** self.stages)

    @property
    def low_w(self) -> int:
        return self.high_w // (2 ** self.stages)


class SyntheticPrecipDataset(Dataset):
    """
    Synthetic paired precipitation / elevation fields.

    High-resolution precipitation depends on:
      - large-scale moisture pattern
      - orographic enhancement from elevation
      - mesoscale variability and noise
    Low-resolution fields are produced by block averaging (8x for 3 stages).
    """

    def __init__(
        self,
        n_days: int,
        high_h: int,
        high_w: int,
        stages: int,
        seed: int = 0,
        for_patches: bool = False,
        patch_size: int = 32,
        patches_per_day: int = 8,
    ):
        self.stages = stages
        self.for_patches = for_patches
        self.patch_size = patch_size
        rng = np.random.default_rng(seed)

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, high_h),
            np.linspace(0.0, 1.0, high_w),
            indexing="ij",
        )
        # Static topography (GTOPO30 analogue)
        elevation = (
            0.55 * np.exp(-((xx - 0.25) ** 2 + (yy - 0.45) ** 2) / 0.03)
            + 0.35 * np.exp(-((xx - 0.70) ** 2 + (yy - 0.60) ** 2) / 0.05)
            + 0.15 * yy
        )
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
        self.elevation = elevation.astype(np.float32)

        factor = 2 ** stages
        samples_x: List[np.ndarray] = []
        samples_y: List[np.ndarray] = []
        samples_e: List[np.ndarray] = []

        for _ in range(n_days):
            phase = rng.uniform(0, 2 * np.pi)
            synoptic = np.maximum(
                0.0,
                6.0
                + 4.0 * np.sin(2 * np.pi * xx + phase)
                + 3.0 * np.cos(2 * np.pi * yy - 0.5 * phase),
            )
            oro = 5.0 * elevation * (1.0 + 0.3 * np.sin(4 * np.pi * xx))
            meso = 1.2 * np.abs(np.sin(10 * np.pi * xx) * np.cos(8 * np.pi * yy))
            noise = rng.normal(0.0, 0.35, size=(high_h, high_w))
            # Zero-inflated wet/dry days (precipitation sparsity)
            wet_mask = rng.random((high_h, high_w)) > 0.35
            high = (synoptic + oro + meso + noise) * wet_mask
            high = np.clip(high, 0.0, None).astype(np.float32)

            low = block_average(high, factor)

            if for_patches:
                for _p in range(patches_per_day):
                    i = rng.integers(0, high_h - patch_size + 1)
                    j = rng.integers(0, high_w - patch_size + 1)
                    y_patch = high[i : i + patch_size, j : j + patch_size]
                    e_patch = elevation[i : i + patch_size, j : j + patch_size]
                    # Coarsen the corresponding high-res patch for LR input
                    x_patch = block_average(y_patch, factor)
                    samples_x.append(x_patch[None, ...])
                    samples_y.append(y_patch[None, ...])
                    samples_e.append(e_patch[None, ...])
            else:
                samples_x.append(low[None, ...])
                samples_y.append(high[None, ...])
                samples_e.append(elevation[None, ...])

        self.x = np.stack(samples_x).astype(np.float32)
        self.y = np.stack(samples_y).astype(np.float32)
        self.e = np.stack(samples_e).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.e[idx]),
            torch.from_numpy(self.y[idx]),
        )


def block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def bilinear_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    return nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)


class SRCNN(nn.Module):
    """
    Augmented SRCNN: bicubic/bilinear upsample of LR precip + HR elevation,
    then 3-layer CNN (9x9 -> 1x1 -> 5x5) as in Dong et al. / DeepSD.
    """

    def __init__(self, scale: int = 2, in_channels: int = 2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, precip_lr: torch.Tensor, elev_hr: torch.Tensor) -> torch.Tensor:
        precip_up = bilinear_upsample(precip_lr, self.scale)
        # Match elevation spatial size to upsampled precip
        if elev_hr.shape[-2:] != precip_up.shape[-2:]:
            elev_hr = nn.functional.interpolate(
                elev_hr, size=precip_up.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([precip_up, elev_hr], dim=1)
        return self.net(x)


class DeepSD(nn.Module):
    """Stacked SRCNNs: each stage 2x, total 8x for stages=3."""

    def __init__(self, stages: int = 3):
        super().__init__()
        self.stages = nn.ModuleList([SRCNN(scale=2) for _ in range(stages)])

    def forward(self, precip_lr: torch.Tensor, elev_hr: torch.Tensor) -> torch.Tensor:
        x = precip_lr
        # Feed elevation at progressively finer scales
        h, w = elev_hr.shape[-2:]
        for i, stage in enumerate(self.stages):
            target_h = precip_lr.shape[-2] * (2 ** (i + 1))
            target_w = precip_lr.shape[-1] * (2 ** (i + 1))
            elev_i = nn.functional.interpolate(
                elev_hr, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            x = stage(x, elev_i)
            x = torch.relu(x)  # precipitation non-negativity soft constraint
        return x


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for precip_lr, elev, precip_hr in loader:
        precip_lr = precip_lr.to(device)
        elev = elev.to(device)
        precip_hr = precip_hr.to(device)
        optimizer.zero_grad()
        pred = model(precip_lr, elev)
        loss = criterion(pred, precip_hr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    losses = []
    for precip_lr, elev, precip_hr in loader:
        precip_lr = precip_lr.to(device)
        elev = elev.to(device)
        precip_hr = precip_hr.to(device)
        pred = model(precip_lr, elev)
        losses.append(criterion(pred, precip_hr).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, y_bcsd = [], [], []
    for precip_lr, elev, precip_hr in loader:
        precip_lr = precip_lr.to(device)
        elev = elev.to(device)
        pred = model(precip_lr, elev)
        # BCSD-like baseline: bilinear upsample only (no learned residual / scaling)
        bcsd = bilinear_upsample(precip_lr, scale=pred.shape[-1] // precip_lr.shape[-1])
        y_true.append(precip_hr.numpy())
        y_pred.append(pred.cpu().numpy())
        y_bcsd.append(bcsd.cpu().numpy())
    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        np.concatenate(y_bcsd, axis=0),
    )


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.ravel()
    yp = y_pred.ravel()
    bias = float(np.mean(yp - yt))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else 0.0
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    # Perkins-style skill on histograms
    bins = np.linspace(0, max(yt.max(), yp.max()) + 1e-6, 51)
    ho, _ = np.histogram(yt, bins=bins, density=True)
    hm, _ = np.histogram(yp, bins=bins, density=True)
    # Convert density * bin width approximation for discrete skill
    skill = float(np.minimum(ho, hm).sum() / max(ho.sum(), 1e-8))
    return {
        "bias": bias,
        "corr": corr,
        "rmse": rmse,
        "mae": mae,
        "skill": skill,
        "mean_true": float(np.mean(yt)),
        "mean_pred": float(np.mean(yp)),
        "median_true": float(np.median(yt)),
        "median_pred": float(np.median(yp)),
        "sd_true": float(np.std(yt)),
        "sd_pred": float(np.std(yp)),
    }


def extreme_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, percentiles: List[float]
) -> pd.DataFrame:
    rows = []
    yt = y_true.ravel()
    yp = y_pred.ravel()
    for p in percentiles:
        thr = np.percentile(yt, p)
        mask = yt >= thr
        if mask.sum() < 10:
            continue
        m = metrics_dict(yt[mask], yp[mask])
        m["percentile"] = p
        m["threshold"] = float(thr)
        m["n_events"] = int(mask.sum())
        rows.append(m)
    return pd.DataFrame(rows)


def save_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bcsd: np.ndarray,
    elev: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (elev[idx, 0], "Elevation (aux)", "terrain"),
        (y_true[idx, 0], "True HR precip", "Blues"),
        (y_pred[idx, 0], "DeepSD precip", "Blues"),
        (y_bcsd[idx, 0], "Bilinear baseline", "Blues"),
        (y_pred[idx, 0] - y_true[idx, 0], "DeepSD error", "RdBu_r"),
        (y_bcsd[idx, 0] - y_true[idx, 0], "Baseline error", "RdBu_r"),
    ]
    for ax, (arr, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("DeepSD demo: stacked SRCNN precipitation downscaling")
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=140)
    plt.close(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = np.random.default_rng(0).choice(y_true.size, size=min(8000, y_true.size), replace=False)
    ax.scatter(y_true.ravel()[sample], y_pred.ravel()[sample], s=4, alpha=0.25, c="steelblue")
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("True precipitation")
    ax.set_ylabel("DeepSD precipitation")
    ax.set_title("Predicted vs true")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_pred_vs_true.png", dpi=140)
    plt.close(fig)


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Grid: LR {cfg.low_h}x{cfg.low_w} -> HR {cfg.high_h}x{cfg.high_w} "
        f"({2 ** cfg.stages}x via {cfg.stages} stacked SRCNN stages)"
    )

    train_ds = SyntheticPrecipDataset(
        cfg.train_days,
        cfg.high_h,
        cfg.high_w,
        cfg.stages,
        seed=cfg.seed,
        for_patches=True,
        patch_size=cfg.patch_size,
        patches_per_day=cfg.patches_per_day,
    )
    # For patch training, HR labels are patches; LR must match stage stack.
    # Rebuild a full-field dataset for val/test evaluation.
    val_ds = SyntheticPrecipDataset(
        cfg.val_days, cfg.high_h, cfg.high_w, cfg.stages, seed=cfg.seed + 1
    )
    test_ds = SyntheticPrecipDataset(
        cfg.test_days, cfg.high_h, cfg.high_w, cfg.stages, seed=cfg.seed + 2
    )

    # Patch dataset uses patch HR size; adjust model training with a patch DeepSD
    # that still stacks 2x stages from coarsened patches.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = DeepSD(stages=cfg.stages).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = []
    best_val = float("inf")
    best_path = cfg.model_dir / "deepsd_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        print(f"Epoch {epoch:03d} | train MSE={tr:.4f} | val MSE={va:.4f}")
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, best_path)

    # Load best and evaluate on test
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred, y_bcsd = predict_all(model, test_loader, device)

    deepsd_m = metrics_dict(y_true, y_pred)
    bcsd_m = metrics_dict(y_true, y_bcsd)
    summary = pd.DataFrame(
        [
            {"model": "DeepSD", **deepsd_m},
            {"model": "BilinearBaseline", **bcsd_m},
        ]
    )
    summary.to_csv(cfg.output_dir / "metrics_summary.csv", index=False)

    extremes = extreme_metrics(y_true, y_pred, percentiles=[90, 95, 99, 99.5])
    extremes.to_csv(cfg.output_dir / "extreme_metrics.csv", index=False)

    # Flat predictions for R diagnostics
    pred_df = pd.DataFrame(
        {
            "y_true": y_true.ravel(),
            "y_pred": y_pred.ravel(),
            "y_baseline": y_bcsd.ravel(),
        }
    )
    # Subsample for manageable R I/O
    if len(pred_df) > 20000:
        pred_df = pred_df.sample(20000, random_state=cfg.seed)
    pred_df.to_csv(cfg.output_dir / "predictions.csv", index=False)

    pd.DataFrame(history).to_csv(cfg.output_dir / "training_history.csv", index=False)
    with open(cfg.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    save_figures(y_true, y_pred, y_bcsd, test_ds.e, cfg.output_dir)

    print("\nTest metrics:")
    print(summary.to_string(index=False))
    print(f"\nSaved artifacts under: {cfg.output_dir.resolve()}")
    print(f"Best checkpoint: {best_path.resolve()}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="DeepSD stacked SRCNN demo pipeline")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--train-days", type=int, default=240)
    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=60)
    p.add_argument("--high-h", type=int, default=64)
    p.add_argument("--high-w", type=int, default=128)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--seed", type=int, default=2017)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    args = p.parse_args()
    return Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        high_h=args.high_h,
        high_w=args.high_w,
        stages=args.stages,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    run(parse_args())
