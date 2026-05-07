# Deep Reinforcement Learning for Energy Systems

This repository is a **reproducible starter project** for exploring how **deep reinforcement learning (DRL)** can be applied to **energy system operation problems**, especially **multi-energy systems** (electric + thermal coupling) under uncertainty.

It includes:
- A **toy multi-energy environment** (Gymnasium-compatible) with electricity/thermal dynamics and time-varying prices.
- A **baseline DRL training script** using `stable-baselines3` (PPO) for quick experiments.
- A **Jupyter notebook workflow** for interactive runs and plots.
- An **R analysis script** illustrating how to summarize controller performance across runs (with example statistics and figures).

> Note: The included environment is intentionally simple so it can run quickly on a laptop. You can progressively replace components (demand model, PV model, thermal model, tariffs, constraints) with more realistic ones.

## Project structure

```
Deep Reinforcement Learning for Energy Systems/
├─ src/
│  ├─ drl_energy_env.py          # Toy multi-energy Gymnasium environment
│  └─ train_sb3.py               # PPO training + evaluation script
├─ notebooks/
│  └─ drl_multi_energy_workflow.ipynb
├─ r/
│  └─ analysis.R                 # Example evaluation + plotting in R
├─ outputs/                      # Generated artifacts (ignored by git)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Setup

### Python (recommended)

Create an environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you already have PyTorch installed, you can keep it and just install the other dependencies.

### R (optional)

Install the packages used by `r/analysis.R`:

```r
install.packages(c("ggplot2","dplyr","readr"))
```

## Quickstart

### Train a PPO agent (toy)

Run from the repository root:

```bash
python -m src.train_sb3 --timesteps 50000 --seed 0
```

Artifacts are saved to `outputs/models/` (ignored by git).

### Open the notebook

```bash
jupyter notebook
```

Then open `notebooks/drl_multi_energy_workflow.ipynb`.

### Run the R analysis example

From the repository root:

```bash
Rscript r/analysis.R
```

This writes synthetic results and plots to `outputs/r/`.

## How the toy environment works

The environment in `src/drl_energy_env.py` is a minimal example of a coupled electric + thermal system:
- **Electric balance**: grid import depends on load, PV generation, battery power, and heat-pump electric consumption.
- **Thermal dynamics**: indoor temperature evolves with simple heat gains (heat pump) and losses (to ambient).
- **Objective**: minimize energy cost (time-varying tariff) while penalizing indoor temperature outside a comfort band.

The observation vector is:
- `soc`: battery state of charge (0–1)
- `indoor_temp`: indoor temperature (°C)
- `pv`: PV generation (kW)
- `elec_load`: electric load (kW)
- `ambient_temp`: ambient temperature (°C)
- `price`: electricity price ($/kWh)

The action vector is:
- `battery_power`: continuous command (scaled to ±battery max kW)
- `heat_pump_power`: continuous command (scaled to heat pump max kW)

## Extending this repository (next steps)

High-value extensions include:
- Adding **real data** (load, PV, weather, tariffs) and scenario splits (train/val/test by time).
- Implementing **constraints** explicitly (action masking, projected actions, constrained RL).
- Comparing against **baselines** (rule-based control, MPC, optimization solvers).
- Adding **robustness tests** (forecast error, rare events, device parameter drift).
- Logging and experiment tracking (seeds, configs, metrics, plots).

## Reference paper

This repository is inspired by the topic described in the guideline file:
- “Deep reinforcement learning as a tool for the analysis and optimization of energy flows in multi-energy systems” (ResearchGate link referenced in `Guidelines_Research_Paper_Review.txt`).

## License

MIT License. See `LICENSE`.


