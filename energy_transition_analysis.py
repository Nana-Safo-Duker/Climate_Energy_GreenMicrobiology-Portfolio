"""
Energy Transition Research Review - Reproducible Analysis Script

This script provides a reusable scaffold for:
1) Loading scenario-level transition data
2) Performing descriptive statistics
3) Running hypothesis tests (t-test)
4) Building illustrative visualizations
5) Exporting clean outputs for reporting
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def ensure_directories() -> None:
    """Create expected project directories if missing."""
    for folder in ["data/raw", "data/processed", "outputs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Keep empty tracked folders with .gitkeep files.
    keep_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "outputs/.gitkeep",
    ]
    for keep_file in keep_files:
        Path(keep_file).touch(exist_ok=True)


def load_data(path: str = "data/raw/transition_scenarios.csv") -> pd.DataFrame:
    """
    Load scenario data or create a synthetic dataset if none exists.
    Expected columns:
    - scenario
    - year
    - renewable_share
    - co2_emissions_mt
    - system_cost_billion_usd
    """
    file_path = Path(path)
    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        np.random.seed(42)
        rows = []
        scenarios = ["baseline", "accelerated_transition"]
        years = np.arange(2025, 2041)
        for scenario in scenarios:
            for year in years:
                renewable_base = 30 + (year - 2025) * (
                    1.2 if scenario == "baseline" else 2.0
                )
                emissions_base = 500 - (year - 2025) * (
                    8 if scenario == "baseline" else 15
                )
                cost_base = 220 + (year - 2025) * (
                    2 if scenario == "baseline" else 1
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "renewable_share": np.clip(
                            np.random.normal(renewable_base, 2.5), 0, 100
                        ),
                        "co2_emissions_mt": max(
                            np.random.normal(emissions_base, 10), 0
                        ),
                        "system_cost_billion_usd": max(
                            np.random.normal(cost_base, 4), 0
                        ),
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped mean, median, std for key variables."""
    metrics = [
        "renewable_share",
        "co2_emissions_mt",
        "system_cost_billion_usd",
    ]
    grouped = df.groupby("scenario")[metrics].agg(["mean", "median", "std"])
    grouped.to_csv("data/processed/descriptive_statistics.csv")
    return grouped


def perform_ttest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Independent t-tests comparing baseline vs accelerated transition.
    This is an illustrative inferential step in the review guideline.
    """
    baseline = df[df["scenario"] == "baseline"]
    accelerated = df[df["scenario"] == "accelerated_transition"]

    results = []
    for var in [
        "renewable_share",
        "co2_emissions_mt",
        "system_cost_billion_usd",
    ]:
        t_stat, p_val = stats.ttest_ind(
            baseline[var], accelerated[var], equal_var=False, nan_policy="omit"
        )
        results.append(
            {"variable": var, "t_statistic": t_stat, "p_value": p_val}
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("data/processed/ttest_results.csv", index=False)
    return result_df


def create_figures(df: pd.DataFrame) -> None:
    """Generate trend and distribution visualizations."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="year", y="renewable_share", hue="scenario", marker="o"
    )
    plt.title("Renewable Share Over Time by Scenario")
    plt.ylabel("Renewable Share (%)")
    plt.tight_layout()
    plt.savefig("outputs/figure_renewable_share_trend.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="year", y="co2_emissions_mt", hue="scenario", marker="o"
    )
    plt.title("CO2 Emissions Over Time by Scenario")
    plt.ylabel("CO2 Emissions (Mt)")
    plt.tight_layout()
    plt.savefig("outputs/figure_emissions_trend.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="scenario", y="system_cost_billion_usd")
    plt.title("System Cost Distribution by Scenario")
    plt.ylabel("System Cost (Billion USD)")
    plt.tight_layout()
    plt.savefig("outputs/figure_cost_boxplot.png", dpi=300)
    plt.close()


def generate_interpretation_notes(
    desc: pd.DataFrame, tests: pd.DataFrame
) -> None:
    """Write compact interpretation notes for the blog/report."""
    lines = []
    lines.append("# Statistical Interpretation Notes")
    lines.append("")
    lines.append("## Descriptive Statistics")
    lines.append("Grouped means, medians, and standard deviations are saved in:")
    lines.append("- data/processed/descriptive_statistics.csv")
    lines.append("")
    lines.append("## Hypothesis Testing")
    lines.append("Welch's t-test compares baseline vs accelerated scenarios.")
    lines.append(
        "A low p-value (< 0.05) suggests statistically meaningful differences."
    )
    lines.append("")
    lines.append("## Quick Result Snapshot")
    for _, row in tests.iterrows():
        lines.append(
            f"- {row['variable']}: t = {row['t_statistic']:.3f}, p = {row['p_value']:.4f}"
        )
    lines.append("")
    lines.append("## Caveat")
    lines.append(
        "Use real transition-model outputs for final claims; "
        "synthetic data is only a workflow template."
    )

    Path("outputs/interpretation_notes.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_directories()
    df = load_data()
    desc = descriptive_statistics(df)
    tests = perform_ttest(df)
    create_figures(df)
    generate_interpretation_notes(desc, tests)

    print("Analysis complete.")
    print("Generated:")
    print("- data/processed/descriptive_statistics.csv")
    print("- data/processed/ttest_results.csv")
    print("- outputs/figure_renewable_share_trend.png")
    print("- outputs/figure_emissions_trend.png")
    print("- outputs/figure_cost_boxplot.png")
    print("- outputs/interpretation_notes.md")


if __name__ == "__main__":
    main()
