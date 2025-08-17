# Climate Change Impacts on Renewable Energy Generation

A comprehensive academic review and reproducible analysis project focused on how climate change reshapes renewable energy potential across technologies and regions. The repository includes Python, R, and notebook workflows to summarize, test, and visualize climate impact estimates.

## Table of Contents

- Overview
- Features
- Project Structure
- Installation
- Usage
- Outputs
- Methodology Highlights
- References
- License

## Overview

This project operationalizes a paper-review workflow around climate impacts on renewable generation. It provides a compact analytical toolkit for comparing technology-level changes and uncertainty while maintaining reproducible outputs suitable for coursework and reporting.

## Features

- Literature-linked climate-energy impact analysis
- Dual implementation in `Python` and `R`
- Interactive notebook for exploratory and narrative workflows
- Exported CSV summaries and publication-ready figures
- Structured outputs for easy reuse in reports and presentations

## Project Structure

```text
.
├── analysis.py
├── analysis.R
├── climate_change_renewables_review.ipynb
├── outputs/
│   ├── global_percent_change_by_technology.png
│   ├── global_percent_change_by_technology_r.png
│   ├── renewable_summary_table.csv
│   ├── renewable_summary_table_r.csv
│   ├── summary_statistics.csv
│   └── summary_statistics_r.csv
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

R dependencies can be installed from an R session as needed.

## Usage

### Python

```bash
python analysis.py
```

### R

```r
source("analysis.R")
```

### Notebook

```bash
jupyter notebook climate_change_renewables_review.ipynb
```

## Outputs

Generated artifacts under `outputs/` include:

- technology-level summary tables (`.csv`)
- descriptive statistics (`.csv`)
- cross-technology impact plots (`.png`)

## Methodology Highlights

- Comparison of projected climate impacts by renewable technology
- Summary statistics and non-parametric significance framing
- Explicit attention to uncertainty sources (model spread, scenario assumptions)
- Reporting designed for transparent scientific communication

## References

- Solaun, K., and Cerda, E. (2019). Climate change impacts on renewable energy generation.
- Gernaat, D. E. H. J. et al. (2021). Climate change impacts on renewable energy supply.

## License

MIT License. See `LICENSE`.

**Last Updated**: August 2025
