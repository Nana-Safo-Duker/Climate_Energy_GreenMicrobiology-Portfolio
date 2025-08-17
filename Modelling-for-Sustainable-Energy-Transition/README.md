# Modelling for Sustainable Energy Transition

A comprehensive research and modeling project for evaluating sustainable energy transition pathways. The repository combines scientific review content with reproducible Python, R, and notebook analysis workflows.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Data Format
- Analysis Workflow
- Outputs
- Reproducibility Notes
- License

## Overview

This project translates energy transition research into an executable analysis framework. It supports both synthetic and user-provided scenario data and emphasizes transparent statistical comparisons and reporting-ready outputs.

## Features

- Scientific review-aligned project setup
- Equivalent analysis workflows in Python and R
- Notebook for interactive narrative analysis
- Scenario comparison with descriptive and inferential statistics
- Flexible data mode (synthetic fallback + real CSV support)

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── outputs/
│   └── .gitkeep
├── energy_transition_analysis.py
├── energy_transition_analysis.R
├── energy_transition_review_notebook.ipynb
├── Research_Paper_Review_Blog_Post.md
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

R setup:

```r
install.packages(c("tidyverse"))
```

## Usage

### Python

```bash
python energy_transition_analysis.py
```

### R

```r
source("energy_transition_analysis.R")
```

### Notebook

```bash
jupyter notebook energy_transition_review_notebook.ipynb
```

## Data Format

When using real data, include `data/raw/transition_scenarios.csv` with columns such as:

- `scenario`
- `year`
- `renewable_share`
- `co2_emissions_mt`
- `system_cost_billion_usd`

## Analysis Workflow

1. Load scenario dataset (real or synthetic fallback).
2. Compute descriptive summaries by scenario and time.
3. Run inferential tests for key transition indicators.
4. Generate plots and export structured outputs.
5. Integrate results into scientific review narrative.

## Outputs

Analysis artifacts are produced in `data/processed/` and `outputs/` for reporting and downstream use.

## Reproducibility Notes

- Keep assumptions explicit when reconstructing scenario data.
- Track data provenance and transformation logic.
- Report uncertainty and methodological limitations clearly.

## License

MIT License. See `LICENSE`.

**Last Updated**: July 2025
