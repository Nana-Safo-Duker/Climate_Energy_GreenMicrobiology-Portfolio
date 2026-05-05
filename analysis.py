"""
Reproducible summary analysis for the review paper
"Climate change impacts on renewable energy generation.
A review of quantitative projections".

This script uses manually curated summary values reported in the literature to:
1. Build a technology-level comparison table.
2. Compute descriptive statistics for absolute and percentage changes.
3. Generate a bar plot of global percentage changes by technology.

The script is intentionally self-contained because the original article's full
underlying gridded data are not included in this repository.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_summary_table() -> pd.DataFrame:
    """Return a summary dataframe from reported global changes."""
    return pd.DataFrame(
        [
            {
                "technology": "Utility-scale PV",
                "absolute_change_ej_per_year": -3.0,
                "percent_change": -0.4,
                "interpretation": (
                    "Minor decrease; solar impacts remain comparatively "
                    "small."
                ),
            },
            {
                "technology": "Rooftop PV",
                "absolute_change_ej_per_year": 0.6,
                "percent_change": 2.0,
                "interpretation": (
                    "Small increase under the higher-warming scenario."
                ),
            },
            {
                "technology": "Concentrated Solar Power",
                "absolute_change_ej_per_year": 7.5,
                "percent_change": 2.2,
                "interpretation": "Modest positive change in global potential.",
            },
            {
                "technology": "Onshore Wind",
                "absolute_change_ej_per_year": -22.1,
                "percent_change": -4.1,
                "interpretation": (
                    "Mean decline, but uncertainty is high across climate "
                    "models."
                ),
            },
            {
                "technology": "Offshore Wind",
                "absolute_change_ej_per_year": -8.6,
                "percent_change": -2.1,
                "interpretation": (
                    "Average decline with strong regional variability."
                ),
            },
            {
                "technology": "Hydropower",
                "absolute_change_ej_per_year": 2.2,
                "percent_change": 6.1,
                "interpretation": (
                    "Net increase globally, but highly uneven across regions."
                ),
            },
            {
                "technology": "First-generation Bioenergy",
                "absolute_change_ej_per_year": 5.8,
                "percent_change": 32.4,
                "interpretation": (
                    "Large increase under CO2 fertilization assumptions."
                ),
            },
            {
                "technology": "Second-generation Bioenergy",
                "absolute_change_ej_per_year": 45.9,
                "percent_change": 38.1,
                "interpretation": (
                    "Largest positive change, but assumption-sensitive."
                ),
            },
        ]
    )


def describe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple descriptive statistics for the numeric columns."""
    return (
        df[["absolute_change_ej_per_year", "percent_change"]]
        .describe()
        .round(2)
    )


def plot_percent_change(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create a bar chart showing reported global percentage changes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "global_percent_change_by_technology.png"

    sns.set_theme(style="whitegrid")
    ordered = df.sort_values("percent_change", ascending=False)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=ordered,
        x="percent_change",
        y="technology",
        hue="technology",
        dodge=False,
        legend=False,
        palette={
            row["technology"]: (
                "#2c7fb8" if row["percent_change"] >= 0 else "#d95f0e"
            )
            for _, row in ordered.iterrows()
        },
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Reported Global Change in Renewable Energy Potential")
    ax.set_xlabel("Percent change (%)")
    ax.set_ylabel("Technology")

    for index, value in enumerate(ordered["percent_change"]):
        x_position = value + 0.8 if value >= 0 else value - 2.4
        ax.text(x_position, index, f"{value:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"

    df = build_summary_table()
    summary_stats = describe_table(df)
    figure_path = plot_percent_change(df, outputs_dir)

    csv_path = outputs_dir / "renewable_summary_table.csv"
    stats_path = outputs_dir / "summary_statistics.csv"

    df.to_csv(csv_path, index=False)
    summary_stats.to_csv(stats_path)

    print("Summary table created:")
    print(df.to_string(index=False))
    print("\nDescriptive statistics:")
    print(summary_stats.to_string())
    print(f"\nSaved figure to: {figure_path}")
    print(f"Saved data table to: {csv_path}")
    print(f"Saved summary statistics to: {stats_path}")


if __name__ == "__main__":
    main()
