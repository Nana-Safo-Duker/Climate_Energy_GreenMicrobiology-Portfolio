"""
Deep learning climate downscaling pipeline (template).

This script demonstrates an end-to-end workflow for supervised temperature
downscaling from low-resolution CMIP6-like fields to high-resolution targets.
It uses synthetic data by default so the pipeline can run without proprietary
datasets, then can be switched to real gridded climate data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    train_samples: int = 300
    val_samples: int = 80
    test_samples: int = 80
    low_h: int = 16
    low_w: int = 16
    upscale: int = 4
    output_dir: Path = Path("outputs")

    @property
    def high_h(self) -> int:
        return self.low_h * self.upscale

    @property
    def high_w(self) -> int:
        return self.low_w * self.upscale


class SyntheticClimateDataset(Dataset):
    """Synthetic paired low-resolution/high-resolution temperature grids."""

    def __init__(self, n: int, low_h: int, low_w: int, upscale: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        high_h, high_w = low_h * upscale, low_w * upscale
        self.x = []
        self.y = []

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, high_h), np.linspace(0.0, 1.0, high_w), indexing="ij"
        )

        for _ in range(n):
            seasonal_phase = rng.uniform(0, 2 * np.pi)
            trend = 2.5 * yy + 1.5 * xx
            wave = np.sin(2 * np.pi * (xx + seasonal_phase)) + np.cos(2 * np.pi * yy)
            mesoscale = 0.4 * np.sin(8 * np.pi * xx) * np.cos(6 * np.pi * yy)
            noise = rng.normal(0, 0.2, size=(high_h, high_w))
            high = 20 + trend + wave + mesoscale + noise

            # Block-average downsampling to emulate low-resolution climate fields.
            high_reshaped = high.reshape(low_h, upscale, low_w, upscale)
            low = high_reshaped.mean(axis=(1, 3))

            self.x.append(low.astype(np.float32)[None, ...])   # 1 x low_h x low_w
            self.y.append(high.astype(np.float32)[None, ...])  # 1 x high_h x high_w

        self.x = np.stack(self.x)
        self.y = np.stack(self.y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class DownscalerCNN(nn.Module):
    """Simple upsampling + CNN refiner for temperature super-resolution."""

    def __init__(self, upscale: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False),
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        losses.append(criterion(pred, y).item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    y_true_all = []
    y_pred_all = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().ravel()
        y_true = y.numpy().ravel()
        y_true_all.append(y_true)
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return float(mae), float(rmse), bias


def main(config: Config) -> None:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SyntheticClimateDataset(
        config.train_samples, config.low_h, config.low_w, config.upscale, seed=config.seed
    )
    val_ds = SyntheticClimateDataset(
        config.val_samples, config.low_h, config.low_w, config.upscale, seed=config.seed + 1
    )
    test_ds = SyntheticClimateDataset(
        config.test_samples, config.low_h, config.low_w, config.upscale, seed=config.seed + 2
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = DownscalerCNN(config.upscale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = config.output_dir / "best_downscaler.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    mae, rmse, bias = evaluate_metrics(model, test_loader, device)

    report_path = config.output_dir / "metrics.txt"
    report = (
        "Evaluation on synthetic test set\n"
        f"MAE:  {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"BIAS: {bias:.4f}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved best model to: {best_path}")
    print(f"Saved metrics report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train climate downscaling CNN.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for model and metrics.")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir),
    )
    main(cfg)
