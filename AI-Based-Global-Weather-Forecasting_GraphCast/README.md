# AI-Based Global Weather Forecasting (GraphCast)

A comprehensive, reproducible starter project for GraphCast-style global weather forecasting evaluation. The project focuses on end-to-end skill analysis by lead time, lightweight significance testing, and publication-ready plotting in both Python and R.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Outputs
- Adapting to Real GraphCast Workflows
- Best Practices
- License

## Overview

This project is designed for paper reviews, coursework, and prototyping of global weather forecast evaluation workflows. It intentionally uses synthetic data so anyone can run the full pipeline locally without external datasets, then swap in ERA5/reanalysis and model forecast data later.

## Features

- Dual-language implementation (`Python` and `R`) for comparable analysis workflows
- End-to-end skill pipeline (data generation -> summary -> t-test -> plotting)
- Notebook-first learning path plus script-first reproducible CLI runs
- Lightweight by default (no large data dependency to get started)
- Clear upgrade path to real GraphCast evaluation pipelines

## Project Structure

```text
.
├── notebooks/
│   └── toy_graphcast_skill_demo.ipynb
├── scripts/
│   ├── skill_demo.py
│   └── skill_demo.R
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

### Python setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### R setup

Install common packages used by the R script:

```r
install.packages(c("dplyr", "ggplot2"))
```

## Usage

### Python script

```bash
python scripts/skill_demo.py --outdir outputs
```

### R script

```r
source("scripts/skill_demo.R")
```

### Notebook

```bash
jupyter notebook notebooks/toy_graphcast_skill_demo.ipynb
```

## Outputs

Typical outputs written under `outputs/`:

- `toy_skill_curve.png`
- `toy_skill_summary.csv`
- `toy_skill_ttests.csv`

## Adapting to Real GraphCast Workflows

To move from synthetic demonstration to production-style analysis:

1. Replace synthetic generator with truth + forecast ingestion.
2. Align data by valid time, variable, and grid definition.
3. Compute per-lead metrics (MAE, RMSE, ACC, regional skill).
4. Add event-oriented metrics for extremes (optional).
5. Keep identical plotting and reporting interfaces for comparability.

## Best Practices

- Use time-aware validation and avoid leakage across lead times.
- Report uncertainty and spread, not only average skill.
- Preserve reproducibility with fixed seeds and explicit configs.
- Track metric definitions exactly when comparing against paper results.

## License

MIT License. See `LICENSE`.
