## AI-Based Global Weather Forecasting (GraphCast) — Starter Repo

This repository is a **lightweight, reproducible starter** for studying and reviewing **GraphCast-style** global weather forecasting papers and for practicing the *evaluation workflow* (skill vs lead time, basic significance testing, plotting).

It is intentionally **data-light**: the included notebook/scripts use **synthetic data** so the repo stays small and runs anywhere. You can later replace the toy generator with **ERA5 / reanalysis + forecast outputs** when you’re ready.

### What’s included

- **`notebooks/toy_graphcast_skill_demo.ipynb`**: end-to-end demo (generate → summarize → t-test → plot).
- **`scripts/skill_demo.py`**: same workflow as a CLI script.
- **`scripts/skill_demo.R`**: same workflow in R.
- **Repo hygiene**: `.gitignore`, `.gitattributes`, `LICENSE`, `requirements.txt`.

### Local-only writing artifacts (not committed)

These files are kept locally but **excluded from git commits/push** by `.gitignore`:

- `Guidelines_Research_Paper_Review.txt`
- `GraphCast_Research_Paper_Review.md`

### Quickstart (Python)

- **Create a virtual environment**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

- **Install dependencies**

```bash
pip install -r requirements.txt
```

- **Run the demo script**

```bash
python scripts/skill_demo.py --outdir outputs
```

This writes:
- `outputs/toy_skill_curve.png`
- `outputs/toy_skill_summary.csv`
- `outputs/toy_skill_ttests.csv`

### Quickstart (R)

The R script expects common packages:
- `ggplot2`
- `dplyr`

Run:

```r
source("scripts/skill_demo.R")
```

### How to adapt this to real GraphCast evaluation

Replace the synthetic error generator with a pipeline that:

- Loads **truth** fields (e.g., ERA5) and **model forecasts** at the same valid times
- Computes per-lead metrics (MAE/RMSE, ACC, region-specific scores)
- (Optional) computes event-based metrics for extremes (threshold skill scores, reliability, etc.)
- Produces the same “skill vs lead” plots and tables

### Repo structure

```
AI-Based Global Weather Forecasting_GraphCast/
  notebooks/
    toy_graphcast_skill_demo.ipynb
  scripts/
    skill_demo.py
    skill_demo.R
  .gitattributes
  .gitignore
  LICENSE
  requirements.txt
  README.md
```

### License

MIT License. See `LICENSE`.
