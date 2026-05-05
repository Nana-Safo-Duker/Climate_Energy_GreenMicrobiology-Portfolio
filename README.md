# Deep Learning for Climate Downscaling (India CMIP6)

This repository provides a structured, reproducible starting point for reviewing and prototyping deep-learning-based climate downscaling workflows, inspired by:

**"Deep Learning for Climate Downscaling: Generating high-resolution gridded temperature projections over India from low-resolution CMIP6 data."**

It includes:
- a scientific review blog post in Markdown,
- a Python training/evaluation pipeline,
- an R-based statistical diagnostics script,
- a detailed Jupyter notebook for interactive experimentation,
- standard GitHub repository configuration files.

---

## Repository Contents

- `Research_Paper_Review_Blog.md`  
  Scientific blog post following the review guideline structure (introduction, background, methods, results, implications, reflection, conclusion).

- `downscaling_pipeline.py`  
  End-to-end Python template for synthetic-data-based climate downscaling model training and evaluation.

- `climate_downscaling_workflow.ipynb`  
  Detailed notebook version of the pipeline with stepwise cells, plots, and adaptation notes for real CMIP6 datasets.

- `downscaling_analysis.R`  
  R script for post-model statistics, paired t-test, and visual diagnostics from prediction outputs.

- `.gitattributes`, `.gitignore`, `LICENSE`, `requirements.txt`  
  Standard repository setup files for collaboration and reproducibility.

---

## Project Goals

1. Reproduce the logic of AI-driven statistical downscaling.
2. Provide a clean scaffold for transitioning from synthetic to real CMIP6 + observed gridded data.
3. Support cross-language analysis (Python for modeling, R for inferential statistics and reporting).
4. Encourage reproducible, documented climate ML experimentation.

---

## Quick Start

### 1) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run Python training pipeline

```bash
python downscaling_pipeline.py --epochs 20 --batch-size 16 --learning-rate 1e-3
```

Expected outputs in `outputs/`:
- `best_downscaler.pt`
- `metrics.txt`

### 4) Run R diagnostics

```bash
Rscript downscaling_analysis.R outputs/predictions.csv outputs
```

If `outputs/predictions.csv` is missing, the script creates a synthetic example automatically.

---

## Recommended Data Layout (for real datasets)

```text
data/
  raw/
    cmip6/
    observations/
  interim/
  processed/
outputs/
models/
```

Suggested variables:
- **Input predictors (coarse):** near-surface temperature and optional auxiliary fields.
- **Target (high-resolution):** observed gridded temperature over India.
- **Dimensions:** `time`, `lat`, `lon`.

---

## Workflow Overview

1. **Ingest and align data**  
   Regrid coarse and high-resolution products onto compatible domains and timelines.

2. **Preprocess**  
   Normalize predictors/targets, handle masks, split into train/val/test by time period.

3. **Train downscaling model**  
   Use convolution-based architecture for spatial super-resolution.

4. **Evaluate quantitatively**  
   Compute MAE, RMSE, and bias; compare against interpolation/statistical baselines.

5. **Diagnose statistically and visually**  
   Use R for mean/median/SD, paired t-test, and residual visualizations.

6. **Document and interpret**  
   Summarize strengths, limitations, uncertainty, and societal relevance.

---

## Reproducibility and Good Practices

- Fix random seeds for deterministic experiments where possible.
- Avoid data leakage by time-aware splitting.
- Track experiment settings (epochs, LR, architecture, preprocessing).
- Report both global and region-wise metrics.
- Include uncertainty caveats for policy-facing interpretations.

---

## Extending This Repository

- Replace synthetic generators with `xarray`/NetCDF loaders.
- Add multi-variable inputs (humidity, pressure, topography, circulation indices).
- Implement model variants (U-Net, ESRGAN-like super-resolution backbones).
- Add uncertainty estimation (ensembles, MC dropout, quantile losses).
- Add benchmark suite (bilinear interpolation, RF/XGBoost, classical statistical downscaling).

---

## Citation

If you use this repository scaffold for coursework or reproducible experiments, cite:

- The original research article under review.
- Any external datasets (CMIP6 products, observational grids).
- Software frameworks used (PyTorch, scikit-learn, R packages).

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
