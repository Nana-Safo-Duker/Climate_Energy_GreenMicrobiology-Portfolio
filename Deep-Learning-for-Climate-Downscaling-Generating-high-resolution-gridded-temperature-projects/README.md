# Deep Learning for Climate Downscaling

A comprehensive deep learning project for climate downscaling from coarse-resolution climate model fields to high-resolution temperature projections. This repository includes Python training workflows, R diagnostics, and notebook-based experimentation tailored to climate research use cases.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Data Guidance
- Workflow
- Extensions
- License

## Overview

This project is built as a reproducible foundation for reviewing and implementing deep-learning climate downscaling methods, including adaptation paths from synthetic demonstration data to real CMIP6 and observational grids.

## Features

- Deep-learning training pipeline in Python
- Statistical diagnostics and reporting in R
- Interactive notebook workflow for stepwise analysis
- Lightweight startup with reproducible defaults
- Clear upgrade path for real climate datasets and models

## Project Structure

```text
.
├── climate_downscaling_workflow.ipynb
├── downscaling_pipeline.py
├── downscaling_analysis.R
├── Research_Paper_Review_Blog.md
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install required R packages in your local R environment as needed.

## Usage

### Python training pipeline

```bash
python downscaling_pipeline.py --epochs 20 --batch-size 16 --learning-rate 1e-3
```

### R diagnostics

```bash
Rscript downscaling_analysis.R outputs/predictions.csv outputs
```

### Notebook workflow

```bash
jupyter notebook climate_downscaling_workflow.ipynb
```

## Data Guidance

For real-world adaptation, structure data with:

- coarse predictors (e.g., CMIP6 near-surface variables)
- high-resolution targets (observational gridded temperature)
- aligned dimensions (`time`, `lat`, `lon`)

Recommended local layout:

```text
data/
  raw/cmip6/
  raw/observations/
  processed/
outputs/
models/
```

## Workflow

1. Ingest and align coarse and high-resolution inputs.
2. Preprocess features and targets with time-safe splits.
3. Train downscaling model.
4. Evaluate MAE/RMSE/bias against baselines.
5. Run diagnostic statistics and visualization.
6. Export reproducible artifacts for reporting.

## Extensions

- Add U-Net or super-resolution backbone variants
- Add uncertainty quantification (ensembles, quantile losses)
- Add multi-variable predictors (humidity, pressure, topography)
- Add benchmark suite (interpolation, tree models, hybrid baselines)

## License

MIT License. See `LICENSE`.
