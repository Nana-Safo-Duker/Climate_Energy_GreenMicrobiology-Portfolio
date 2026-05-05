# Reproducible summary analysis for:
# "Climate change impacts on renewable energy generation.
#  A review of quantitative projections"
# Solaun & Cerda (2019), Renewable and Sustainable Energy Reviews, 116, 109415.
#
# Uses manually curated summary values reported in the literature.
# Produces: comparison table (CSV), descriptive statistics (CSV), bar chart (PNG).
# All outputs are written to <script_dir>/outputs/.

# ── Resolve output directory relative to this script ─────────────────────────
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}
output_dir <- file.path(script_dir, "outputs")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ── Data ─────────────────────────────────────────────────────────────────────
summary_df <- data.frame(
  technology = c(
    "Utility-scale PV",
    "Rooftop PV",
    "Concentrated Solar Power",
    "Onshore Wind",
    "Offshore Wind",
    "Hydropower",
    "First-generation Bioenergy",
    "Second-generation Bioenergy"
  ),
  absolute_change_ej_per_year = c(
    -3.0, 0.6, 7.5, -22.1, -8.6, 2.2, 5.8, 45.9
  ),
  percent_change = c(
    -0.4, 2.0, 2.2, -4.1, -2.1, 6.1, 32.4, 38.1
  ),
  interpretation = c(
    "Minor decrease; solar impacts remain comparatively small.",
    "Small increase under the higher-warming scenario.",
    "Modest positive change in global potential.",
    "Mean decline, but uncertainty is high across climate models.",
    "Average decline with strong regional variability.",
    "Net increase globally, but highly uneven across regions.",
    "Large increase under CO2 fertilization assumptions.",
    "Largest positive change, but assumption-sensitive."
  ),
  stringsAsFactors = FALSE
)

# ── Descriptive statistics ────────────────────────────────────────────────────
summary_stats <- data.frame(
  metric = c("mean", "median", "sd", "min", "max"),
  absolute_change_ej_per_year = c(
    mean(summary_df$absolute_change_ej_per_year),
    median(summary_df$absolute_change_ej_per_year),
    sd(summary_df$absolute_change_ej_per_year),
    min(summary_df$absolute_change_ej_per_year),
    max(summary_df$absolute_change_ej_per_year)
  ),
  percent_change = c(
    mean(summary_df$percent_change),
    median(summary_df$percent_change),
    sd(summary_df$percent_change),
    min(summary_df$percent_change),
    max(summary_df$percent_change)
  )
)

# ── Write CSVs ────────────────────────────────────────────────────────────────
write.csv(
  summary_df,
  file.path(output_dir, "renewable_summary_table_r.csv"),
  row.names = FALSE
)
write.csv(
  summary_stats,
  file.path(output_dir, "summary_statistics_r.csv"),
  row.names = FALSE
)

# ── Figure ────────────────────────────────────────────────────────────────────
ordered_df <- summary_df[order(summary_df$percent_change), ]
bar_colors <- ifelse(ordered_df$percent_change >= 0, "#2c7fb8", "#d95f0e")

figure_path <- file.path(
  output_dir, "global_percent_change_by_technology_r.png"
)
png(figure_path, width = 1200, height = 700, res = 150)
par(mar = c(5, 14, 4, 3))
bar_positions <- barplot(
  ordered_df$percent_change,
  horiz    = TRUE,
  col      = bar_colors,
  las      = 1,
  names.arg = ordered_df$technology,
  xlab     = "Percent change (%)",
  main     = "Reported Global Change in Renewable Energy Potential",
  xlim     = c(
    min(ordered_df$percent_change) - 5,
    max(ordered_df$percent_change) + 8
  )
)
abline(v = 0, lwd = 1)
text(
  x      = ordered_df$percent_change +
    ifelse(ordered_df$percent_change >= 0, 2, -2),
  y      = bar_positions,
  labels = paste0(round(ordered_df$percent_change, 1), "%"),
  cex    = 0.8
)
dev.off()

# ── Console output ────────────────────────────────────────────────────────────
cat("Summary table:\n")
print(summary_df)
cat("\nDescriptive statistics:\n")
print(summary_stats)
cat(sprintf("\nFigure saved to: %s\n", figure_path))
