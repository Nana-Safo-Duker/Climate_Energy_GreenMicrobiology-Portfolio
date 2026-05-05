# Modelling for Sustainable Energy Transition

Scientific review and reproducible analysis toolkit based on the paper:
**Modelling for sustainable energy transition**  
Source: https://www.researchgate.net/publication/390549686_Modelling_for_sustainable_energy_transition

This repository includes:

- A complete scientific blog post aligned with your assignment guideline.
- A Python workflow for descriptive statistics, t-tests, and figure generation.
- An R workflow with equivalent statistical analysis and plots.
- A Jupyter notebook for interactive exploration and reporting.
- Standard GitHub repository scaffolding files.

## Project Structure

```text
.
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── Guidelines_Research_Paper_Review.txt
├── Research_Paper_Review_Blog_Post.md
├── energy_transition_analysis.py
├── energy_transition_analysis.R
├── energy_transition_review_notebook.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
```

## Files Overview

- `Research_Paper_Review_Blog_Post.md`  
  Full scientific blog post following the required structure: Introduction, Background, Methodology, Results, Discussion, Reflection, Conclusion, and References.

- `Research_Paper_Review_Blog.md`  
  Alternate local copy of the blog post kept in the project and excluded from push.

- `energy_transition_analysis.py`  
  End-to-end reproducible Python script:
  - Creates project directories
  - Loads or generates scenario data
  - Computes mean/median/standard deviation
  - Performs Welch t-tests
  - Saves figures and interpretation notes

- `energy_transition_analysis.R`  
  Equivalent R pipeline using `tidyverse` for data handling and plotting.

- `energy_transition_review_notebook.ipynb`  
  Interactive notebook for narrative + code workflow and figure generation.

## Quick Start

### 1) Clone repository

```bash
git clone <your-repo-url>
cd "Modelling for sustainable energy transition"
```

### 2) Set up Python environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Run Python analysis

```bash
python energy_transition_analysis.py
```

Generated outputs will appear in:

- `data/processed/`
- `outputs/`

### 4) Run Jupyter notebook

```bash
jupyter notebook energy_transition_review_notebook.ipynb
```

### 5) Run R analysis

Install required packages in R (if missing):

```r
install.packages(c("tidyverse"))
```

Then run:

```r
source("energy_transition_analysis.R")
```

## Data Notes

The scripts support two modes:

1. **Real-data mode:** If `data/raw/transition_scenarios.csv` exists, it is used directly.
2. **Template mode:** If no data file exists, synthetic data is generated so the workflow can be tested end-to-end.

Expected CSV columns:

- `scenario`
- `year`
- `renewable_share`
- `co2_emissions_mt`
- `system_cost_billion_usd`

## Suggested Academic Workflow

1. Read and annotate the target paper.
2. Use `Research_Paper_Review_Blog_Post.md` as a structured base.
3. Replace template/synthetic data with extracted or reconstructed data.
4. Run Python, notebook, and/or R pipelines.
5. Integrate generated plots into your final blog/report with citations and interpretation.

## Reproducibility and Ethics

- Keep assumptions explicit when reconstructing paper results.
- Document data provenance and transformation steps.
- Distinguish observed findings from model assumptions.
- Report uncertainty and limitations transparently.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
