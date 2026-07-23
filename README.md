# Climate, Energy & Green Microbiology Portfolio

Multi-project laboratory of climate intelligence, energy systems analytics, and sustainability-focused machine learning by Nana Safo-Duker. Each folder is a self-contained workflow targeting a distinct forecasting, optimization, or transition-planning problem—from extreme-weather risk and global weather AI to energy demand/load/price modeling, climate-driven renewable optimization, diffusion ensembles, statistical downscaling, and foundation-model evaluation demos.

This README provides the cross-project narrative: structure, shared tooling, reproducibility expectations, and project-by-project context.

## Table of Contents
- [About](#about)
- [Portfolio Overview](#portfolio-overview)
- [Visualizations](#visualizations)
- [Repository Layout](#repository-layout)
- [Technology Stack and Tooling Matrix](#technology-stack-and-tooling-matrix)
- [Shared Setup Workflow](#shared-setup-workflow)
- [Workflow Blueprint](#workflow-blueprint)
- [Project Capsules](#project-capsules)
- [Data Sources and Governance](#data-sources-and-governance)
- [Testing and Validation Hooks](#testing-and-validation-hooks)
- [Extensibility Playbook](#extensibility-playbook)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Contact](#contact)

## About
**Description:** AI/ML portfolio spanning climate prediction, energy forecasting, renewable optimization, and sustainable transition modeling—including GraphCast-style evaluation, diffusion ensembles, DeepSD super-resolution downscaling, and ClimaX-inspired foundation-model demos.  
**Repository:** [Climate_Energy_GreenMicrobiology-Portfolio](https://github.com/Nana-Safo-Duker/Climate_Energy_GreenMicrobiology-Portfolio)  
**Website:** https://nana-safo-duker.github.io/  
**Topics:** climate-ai, energy-ml, renewable-energy, weather-forecasting, load-forecasting, electricity-pricing, sustainability, graph-neural-networks, diffusion-models, climate-downscaling, foundation-models, forecasting.

> The portfolio title retains “Green Microbiology” for branding continuity; there is no separate microbiology project folder in this repository.

## Portfolio Overview
- **Domains represented:** climate and weather forecasting, extreme-event classification, electricity price forecasting, load/demand forecasting, renewable resource optimization, climate-impact assessment on renewables, deep reinforcement learning for energy systems, transition scenario modeling, probabilistic diffusion ensembles, statistical/DL climate downscaling, and foundation-model evaluation.
- **Core methods:** tree ensembles, time-series forecasting, deep learning (CNN/Transformer-style motifs), graph-based weather evaluation, reinforcement learning (PPO), probabilistic verification (RMSE/MAE/CRPS/SSR), rare-event classification metrics, and explainable ML diagnostics.
- **Languages and runtimes:** Python 3.10+, R 4.x, Jupyter Notebooks; nearly every project ships parallel Python + R + notebook paths.
- **Deliverables:** reproducible notebooks, CLI/scripts, diagnostic plots (`assets/overview.png` per project), summary CSVs, and project-level documentation (often with paper-review companions).
- **Operational model:** **13** standalone climate/energy project folders. Most demos are **synthetic-first** with a documented upgrade path to ERA5, WeatherBench, CMIP6, PRISM, or market datasets.

## Visualizations

Portfolio-level figures in `assets/` summarize themes across all thirteen climate/energy projects. Each project folder also includes its own `assets/overview.png`.

### Cross-project analytical overview

Six panels spanning weather skill vs lead time, climate→renewables percent change, load/demand profiles, extreme-event ROC, transition renewable share (2020–2050), and downscaling fit:

![Portfolio cross-project visual overview](assets/portfolio_overview.png)

### Project theme map (climate & energy only)

Constellation map of the **13 climate/energy folders** across three lanes—**Climate & Weather AI** (7), **Energy Forecasting** (4), and **Systems & Transition** (2). This figure is unique to this repository—not shared with the biomedical or epidemiology portfolios:

![Portfolio project theme map](assets/project_theme_map.png)

> Root figures are illustrative summaries. Open any project folder for runnable pipelines and project-specific plots.

## Repository Layout

Top-level directories (each is an independent project; names match the repository exactly, including spaces where present):

```text
Climate_Energy_GreenMicrobiology-Portfolio/
├── assets/                                      # Portfolio-level visualizations
│   ├── portfolio_overview.png
│   └── project_theme_map.png
├── AI-Based-Global-Weather-Forecasting_GraphCast/
├── Climate-Data-for-Renewable-Energy-Optimization/
├── Climate-change-impacts-on-renewable-energy-generation/
├── ClimaX A foundation model for weather and climate/
├── Deep-Learning-for-Climate-Downscaling-Generating-high-resolution-gridded-temperature-projects/
├── Deep-Reinforcement-Learning-for-Energy-Systems/
├── DeepSD Generating High Resolution Climate Change Projections through Single Image Super-Resolution/
├── Energy-Load-Forecasting-Using-Machine-Learning/
├── Forecasting-Electricity-Price-IMachine-Learning-Models-and-Strategies/
├── ML-based-Energy-demand-prediction/
├── ML-for-Extreme-Weather-Event-Prediction/
├── Modelling-for-Sustainable-Energy-Transition/
├── Weather Forecasting with Diffusion models/
└── README.md
```

Each project typically includes:
- `data/` or dataset placeholders / ingestion scripts (often synthetic by default)
- `notebooks/` for EDA + experiment walkthroughs
- `src/` or `scripts/` (and often root `.py` / `.R` pipelines) for modular execution
- `results/`, `outputs/`, `figures/`, `assets/` (README overview figure), or model checkpoints
- Project-specific `README.md`, `requirements.txt`, and repository hygiene files (`.gitignore`, `.gitattributes`, `LICENSE`)

## Technology Stack and Tooling Matrix

| Layer | Tooling | Where Used |
|---|---|---|
| Languages | Python, R | Cross-project implementation and reproducibility |
| Environments | `requirements.txt`, virtual environments, optional Conda | Deterministic setup by project |
| Notebooks | Jupyter (`.ipynb`) | Interactive analysis and paper-review walkthroughs |
| Classical ML | scikit-learn, Ridge/HGB, Random Forest, R (`ranger`, `randomForest`, tidymodels-style) | Load, demand, price, renewable, and extreme-event baselines |
| Deep learning / RL | PyTorch (DeepSD SRCNN), Gymnasium + Stable-Baselines3 (PPO), Transformer/diffusion motifs | Downscaling, control, and weather-AI demos |
| Weather / climate eval | Skill curves (MAE/RMSE/ACC), CRPS/SSR, ROC/PR-AUC, paired *t*-tests | GraphCast-style, diffusion, ClimaX, extreme-event projects |
| Visualization | matplotlib, seaborn, ggplot2 | Diagnostic reporting and README assets |
| Serialization | CSV metrics, optional `.pkl` / checkpoints, notebook outputs | Experiment records and model artifacts |
| QA | Fixed seeds, Python↔R parity checks, baseline metrics | Regression anchors and reproducibility confirmation |

## Shared Setup Workflow

Clone once to access all projects:

```bash
git clone https://github.com/Nana-Safo-Duker/Climate_Energy_GreenMicrobiology-Portfolio.git
cd Climate_Energy_GreenMicrobiology-Portfolio
```

Then:
1. Enter one project folder and read its local `README.md`.
2. Create or activate an environment (`python -m venv .venv` or project-specific setup).
3. Install dependencies from `requirements.txt` (and R packages noted in the project README).
4. Run notebooks (`jupyter notebook`) or scripts (`python ...`, `Rscript ...`).
5. Save generated artifacts under that project's `outputs/`, `results/`, or `models/` structure.

> Note: Many projects ship synthetic or placeholder data so pipelines run without large downloads. Replace with licensed or institution-approved datasets (ERA5, WeatherBench, CMIP6, market series, etc.) for production-quality analysis.

## Workflow Blueprint
- **Problem selection:** choose forecasting, optimization, downscaling, foundation-model evaluation, or transition domain.
- **Environment provisioning:** install project-specific Python/R dependencies.
- **Notebook rehearsal:** run exploratory notebooks for feature engineering and baseline metrics.
- **Script automation:** switch to CLI scripts for repeatable training/inference.
- **Evaluation pass:** generate and review RMSE/MAE/MAPE/AUC/CRPS (as applicable), plus plots.
- **Reporting/export:** package figures, metrics, and model artifacts for publication or stakeholder review.

## Project Capsules

### 1) Deep-Reinforcement-Learning-for-Energy-Systems
**Problem statement:** optimize multi-energy system operation under uncertainty with deep reinforcement learning.  
**Highlights:** Gymnasium environment; PPO via Stable-Baselines3; Python training plus R diagnostics; battery/heat-pump and tariff-aware control motifs.  
**Use cases:** policy optimization, adaptive control, cost vs thermal-comfort tradeoff tuning.

### 2) ML-for-Extreme-Weather-Event-Prediction
**Problem statement:** classify rare extreme-weather events (heavy precip, high temperature, extreme wind) from atmospheric signals.  
**Highlights:** percentile-based labeling; Logistic Regression + Random Forest; PR-AUC/ROC-AUC/F1/Brier; blocked time splits; Python + R + notebook.  
**Use cases:** early-warning prototypes, resilience planning, calibration-aware risk analytics education.

### 3) Forecasting-Electricity-Price-IMachine-Learning-Models-and-Strategies
**Problem statement:** forecast electricity price indices with competing ML strategies and exogenous drivers.  
**Highlights:** lag/rolling features; chronological CV; Python + R pipelines; weather/load/fuel/regime-style exogenous features.  
**Use cases:** market operations research, procurement strategy demos, volatility-aware planning.

### 4) Climate-Data-for-Renewable-Energy-Optimization
**Problem statement:** forecast renewable generation from climate/meteo drivers with time-series-aware backtesting.  
**Highlights:** Ridge baseline vs tuned tree models; rolling/expanding validation; MAE tracking; Python + R.  
**Use cases:** climate-informed yield forecasting, operational renewable planning, meteo-driven capacity analytics.

### 5) Energy-Load-Forecasting-Using-Machine-Learning
**Problem statement:** short-term energy load forecasting with cyclical and lag/rolling features.  
**Highlights:** linear + tree baselines; MAE/RMSE/MAPE; chronological CV; bilingual Python/R workflows.  
**Use cases:** grid balancing support, dispatch prototyping, capacity-planning exercises.

### 6) AI-Based-Global-Weather-Forecasting_GraphCast
**Problem statement:** GraphCast-*inspired* global weather forecast **evaluation** (skill vs lead time)—not full GraphCast training.  
**Highlights:** synthetic skill curves; MAE/RMSE/ACC path; significance bands; Python + R + notebook; upgrade path to ERA5.  
**Use cases:** large-scale forecast evaluation prototyping, paper-review coursework, atmospheric ML benchmarking scaffolds.

### 7) Modelling-for-Sustainable-Energy-Transition
**Problem statement:** compare decarbonization pathways with scenario-centric statistics on renewable share, emissions, and system cost.  
**Highlights:** descriptive + inferential scenario analytics; Python + R + notebook; synthetic or CSV scenario tables.  
**Use cases:** transition roadmapping, scenario comparison, policy/technology support demos.

### 8) Deep-Learning-for-Climate-Downscaling-Generating-high-resolution-gridded-temperature-projects
**Problem statement:** deep learning climate downscaling from coarse to high-resolution temperature grids.  
**Highlights:** Python training pipeline; R diagnostics (MAE/RMSE/bias); notebook walkthrough; path to CMIP6 + observations.  
**Use cases:** localized climate risk modeling, regional planning, adaptation studies.

### 9) Climate-change-impacts-on-renewable-energy-generation
**Problem statement:** synthesize and analyze how climate-change signals affect renewable generation potential by technology.  
**Highlights:** literature-linked impact summaries; non-parametric significance; Python + R + notebook; CSV/figure exports.  
**Use cases:** cross-technology impact comparison, climate stress-testing narratives, long-term investment discussion support.

### 10) ML-based-Energy-demand-prediction
**Problem statement:** predict utility-scale energy demand with lag/rolling features and classical ML.  
**Highlights:** Ridge / HistGradientBoosting CLI; chronological split; MAE/RMSE; Python + R + notebook.  
**Use cases:** utility forecasting demos, demand-response design exercises, infrastructure planning scaffolds.

### 11) Weather Forecasting with Diffusion models
**Problem statement:** educational companion to continuous ensemble weather forecasting with diffusion (Andrae et al., 2024).  
**Highlights:** lead-time conditioning; correlated noise; ARCI-style hybrids; RMSE/CRPS/SSR; rich `assets/` gallery; Python CLI + R + notebook (synthetic analogue—not full U-Net diffusion training).  
**Use cases:** high-temporal-resolution ensemble demos, uncertainty quantification education, renewable weather-risk support.

### 12) DeepSD Generating High Resolution Climate Change Projections through Single Image Super-Resolution
**Problem statement:** stacked SRCNN super-resolution for statistical climate downscaling (Vandal et al., 2017 / DeepSD).  
**Highlights:** PyTorch stacked SRCNN (8×) with elevation channel; bilinear/BCSD-like baseline; extremes metrics; R paired *t*-tests; notebook.  
**Use cases:** local climate-risk products, extreme precipitation downscaling education, ESM ensemble post-processing prototypes.

### 13) ClimaX A foundation model for weather and climate
**Problem statement:** research-review and synthetic evaluation companion to ClimaX (Nguyen et al., ICML 2023).  
**Highlights:** variable-tokenization concepts; forecast skill vs lead; ClimateBench-style scores; downscaling RMSE; scaling curves; Python + R + Jupyter (educational—not official microsoft/ClimaX training).  
**Use cases:** multi-task weather evaluation demos, ClimateBench-style projection transfer exercises, Earth-system ML prototyping.

## Data Sources and Governance
- Use only datasets with clear license terms and permitted reuse.
- Keep sensitive/private data and large raw climate archives outside the repository.
- Store credentials and paths in `.env` or secure config (gitignored).
- Document provenance, preprocessing assumptions, and temporal coverage for each project.
- Treat synthetic scaffolds as educational unless a project README explicitly documents otherwise.

## Testing and Validation Hooks
- Re-run notebooks/scripts with fixed seeds where possible.
- Prefer Python↔R parity checks on shared metrics tables.
- Track baseline metrics per project (e.g., MAE/RMSE/MAPE/AUC/CRPS).
- Version key plots (skill curves, residual analysis, ROC/PR curves, confusion matrices).
- Add smoke tests for data loading and training pipeline integrity.

## Extensibility Playbook
- Add new projects with the same folder contract (`data`, `notebooks`, `scripts/src`, `outputs`/`results`, `assets/overview.png`, docs).
- Promote reusable utilities into a shared `common/` package when overlap grows.
- Add model cards and dataset cards to improve transparency.
- Introduce CI checks for linting, environment validation, and notebook smoke tests.
- Regenerate root `assets/project_theme_map.png` when new thematic folders land.

## Contributing
1. Create a branch: `git checkout -b feature/<name>`
2. Keep edits scoped to a single project unless refactoring shared utilities.
3. Update both project-level docs and this root README when behavior or inventory changes.
4. Validate runs and include metrics/plots in PR descriptions.
5. Open a PR with reproducibility notes and data/license context.

## Roadmap
- Standardize dependency/environment files across all project folders.
- Add consistent model evaluation reports and comparison dashboards.
- Introduce lightweight CI for notebook/script smoke tests.
- Publish tagged portfolio releases by theme (weather AI, demand/load, market, transition, downscaling).
- Expand documentation on data lineage and model governance.
- Grow selected demos from synthetic scaffolds toward WeatherBench/ERA5/CMIP6-backed evaluations.

## Contact
For collaboration, consulting, or demo requests, open an issue or connect via:
https://nana-safo-duker.github.io

---

Also see related repositories under Nana Safo-Duker's GitHub profile for companion [computational biomedical](https://github.com/Nana-Safo-Duker/Computational_Biomedical_Research-Portfolio) and [infectious-disease epidemiology](https://github.com/Nana-Safo-Duker/Infectious-Disease-Modelling_Epidemiology-Portfolio) portfolios.

**Last Updated**: July 2026
