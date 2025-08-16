# Climate Change Impacts on Renewable Energy Generation

A structured academic review package for:

> Solaun, K., & Cerda, E. (2019). *Climate change impacts on renewable energy generation.
> A review of quantitative projections.* Renewable and Sustainable Energy Reviews, 116, 109415.
> DOI: [10.1016/j.rser.2019.109415](https://doi.org/10.1016/j.rser.2019.109415)

---

## Repository Structure

```text
.
├── Guidelines_Research_Paper_Review.txt   # Assignment rubric (do not modify)
├── scientific_blog_post.md                # Full scientific blog post (Steps 1–9)
├── analysis.py                            # Python summary analysis script
├── analysis.R                             # R version of the same analysis
├── climate_change_renewables_review.ipynb # Jupyter notebook (interactive)
├── requirements.txt                       # Pinned Python dependencies (Python 3.11)
├── LICENSE                                # MIT licence
├── .gitattributes                         # Line-ending normalisation rules
├── .gitignore                             # Excluded files
└── outputs/                               # Generated files (tracked in git)
    ├── global_percent_change_by_technology.png
    ├── global_percent_change_by_technology_r.png
    ├── renewable_summary_table.csv
    ├── renewable_summary_table_r.csv
    ├── summary_statistics.csv
    └── summary_statistics_r.csv
```

---

## Paper Summary

The reviewed paper addresses a single focused question: **how does future climate change
alter the generation potential of renewable energy technologies?**

Key conclusions:

| Technology | Global % change | Uncertainty |
| --- | --- | --- |
| Second-generation Bioenergy | +38.1 % | High (CO₂ fertilisation assumption) |
| First-generation Bioenergy | +32.4 % | High (CO₂ fertilisation assumption) |
| Hydropower | +6.1 % | High (regionally mixed) |
| Concentrated Solar Power | +2.2 % | Low–moderate |
| Rooftop PV | +2.0 % | Low |
| Utility-scale PV | −0.4 % | Low |
| Offshore Wind | −2.1 % | High (model spread) |
| Onshore Wind | −4.1 % | High (model spread) |

Values are sourced from the companion study (Gernaat et al., 2021, *Nature Climate Change*)
which operationalises the framework reviewed by Solaun & Cerda (2019).

---

## Why This Matters

Renewable energy is usually framed as a *solution* to climate change. This paper shows that
renewable resources are themselves **climate-sensitive**: solar irradiance, wind speed,
precipitation, and crop productivity all change under warming, affecting generation potential.

Long-term energy infrastructure planning must therefore account for:

- changes in physical resource availability across technologies,
- strong regional variability (some basins gain hydropower, others lose it),
- large model uncertainty, especially for wind and hydropower,
- assumption sensitivity for bioenergy (CO₂ fertilisation).

---

## Reproducible Analysis

The Python, R, and notebook files use a **self-contained summary dataset** curated from
reported global-scale results in the literature. They do not reproduce the paper's original
full climate-model pipeline (which requires gridded GCM outputs not included here).

### What the analysis produces

- A technology-level comparison table (CSV)
- Descriptive statistics — mean, median, standard deviation, min, max (CSV)
- A horizontal bar chart of global percentage change by technology (PNG)

---

## Quick Start

### Python

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python analysis.py
```

### Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open `climate_change_renewables_review.ipynb`.

### R

Open an R session in this directory and run:

```r
source("analysis.R")
```

Outputs are written to `outputs/`.

---

## Methodology Notes

| Aspect | Detail |
| --- | --- |
| Climate scenarios | RCP 2.6 (mitigation) and RCP 6.0 (baseline warming) |
| Climate models | Four GCMs: GFDL-ESM2M, HadGEM2-ES, IPSL-CM5A-LR, MIROC5 |
| Technologies assessed | Utility PV, Rooftop PV, CSP, Onshore Wind, Offshore Wind, Hydro, 1G & 2G Bioenergy |
| Statistical test | Wilcoxon signed-rank test (historical vs. future period) |
| IAM integration | IMAGE model with cost–supply curves under SSP2 scenarios |
| Key uncertainty | GCM selection drives the largest spread; CO₂ fertilisation drives bioenergy range |

---

## Academic Use

This repository is suitable for:

- research paper review assignments (follows the rubric in `Guidelines_Research_Paper_Review.txt`),
- reproducible figure demonstrations in coursework,
- introduction to climate-energy data analysis with Python and R,
- presentation and portfolio preparation.

---

## References

1. Solaun, K., & Cerda, E. (2019). *Climate change impacts on renewable energy generation.
   A review of quantitative projections.* Renewable and Sustainable Energy Reviews, 116, 109415.
   [doi:10.1016/j.rser.2019.109415](https://doi.org/10.1016/j.rser.2019.109415)

2. Gernaat, D. E. H. J., de Boer, H. S., Daioglou, V., Yalew, S. G., Müller, C., &
   van Vuuren, D. P. (2021). *Climate change impacts on renewable energy supply.*
   Nature Climate Change, 11, 119–125.
   [doi:10.1038/s41558-020-00949-9](https://doi.org/10.1038/s41558-020-00949-9)
