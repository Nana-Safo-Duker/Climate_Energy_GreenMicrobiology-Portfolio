# DeepSD precipitation downscaling diagnostics in R
#
# Post-training statistical evaluation aligned with the paper's metrics:
#   bias, correlation, RMSE, distributional summaries, paired t-test,
#   and extreme-event threshold analysis.
#
# Expected input CSV schema (from deepsd_pipeline.py):
#   y_true, y_pred, y_baseline

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- ifelse(length(args) >= 1, args[[1]], "outputs/predictions.csv")
output_dir <- ifelse(length(args) >= 2, args[[2]], "outputs")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (!file.exists(input_path)) {
  message("Input file not found: ", input_path)
  message("Creating a synthetic DeepSD-style example dataset.")
  set.seed(2017)
  n <- 12000
  y_true <- pmax(0, rnorm(n, mean = 3.2, sd = 2.8))
  y_pred <- pmax(0, y_true + rnorm(n, mean = 0.05, sd = 1.1))
  y_baseline <- pmax(0, y_true + rnorm(n, mean = 0.20, sd = 1.6))
  demo <- data.frame(y_true = y_true, y_pred = y_pred, y_baseline = y_baseline)
  if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)
  write_csv(demo, "outputs/predictions.csv")
  input_path <- "outputs/predictions.csv"
}

df <- read_csv(input_path, show_col_types = FALSE)

required_cols <- c("y_true", "y_pred")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

if (!"y_baseline" %in% names(df)) {
  df$y_baseline <- df$y_pred
}

df <- df %>%
  mutate(
    error_deepsd = y_pred - y_true,
    error_baseline = y_baseline - y_true
  )

metric_block <- function(true_vals, pred_vals, model_name) {
  err <- pred_vals - true_vals
  tibble(
    model = model_name,
    n = length(true_vals),
    mean_true = mean(true_vals),
    median_true = median(true_vals),
    sd_true = sd(true_vals),
    mean_pred = mean(pred_vals),
    median_pred = median(pred_vals),
    sd_pred = sd(pred_vals),
    bias = mean(err),
    mae = mean(abs(err)),
    rmse = sqrt(mean(err^2)),
    corr = suppressWarnings(cor(true_vals, pred_vals))
  )
}

stats <- bind_rows(
  metric_block(df$y_true, df$y_pred, "DeepSD"),
  metric_block(df$y_true, df$y_baseline, "BilinearBaseline")
)

# Paired t-test: DeepSD predictions vs observations
tt_deepsd <- t.test(df$y_pred, df$y_true, paired = TRUE)
# Compare absolute errors DeepSD vs baseline
tt_err <- t.test(abs(df$error_deepsd), abs(df$error_baseline), paired = TRUE)

stats_path <- file.path(output_dir, "r_statistics_summary.csv")
write_csv(stats, stats_path)

t_test_path <- file.path(output_dir, "r_paired_t_test.txt")
sink(t_test_path)
cat("=== Paired t-test: DeepSD y_pred vs y_true ===\n")
print(tt_deepsd)
cat("\n=== Paired t-test: |DeepSD error| vs |Baseline error| ===\n")
print(tt_err)
sink()

# Extreme-event analysis (paper Figure 5 style)
percentiles <- c(90, 95, 99)
extreme_rows <- list()
for (p in percentiles) {
  thr <- quantile(df$y_true, probs = p / 100, names = FALSE)
  sub <- df %>% filter(y_true >= thr)
  if (nrow(sub) < 20) next
  extreme_rows[[length(extreme_rows) + 1]] <- metric_block(
    sub$y_true, sub$y_pred, paste0("DeepSD_p", p)
  ) %>%
    mutate(percentile = p, threshold = thr)
  extreme_rows[[length(extreme_rows) + 1]] <- metric_block(
    sub$y_true, sub$y_baseline, paste0("Baseline_p", p)
  ) %>%
    mutate(percentile = p, threshold = thr)
}
extreme_df <- bind_rows(extreme_rows)
extreme_path <- file.path(output_dir, "r_extreme_metrics.csv")
write_csv(extreme_df, extreme_path)

# Scatter: DeepSD
scatter_plot <- ggplot(df, aes(x = y_true, y = y_pred)) +
  geom_point(alpha = 0.15, color = "#1f4e79", size = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "#b22222", linetype = "dashed") +
  labs(
    title = "DeepSD: Predicted vs True Precipitation",
    x = "True precipitation",
    y = "Predicted precipitation"
  ) +
  theme_minimal(base_size = 12)

# Error histogram comparison
err_long <- bind_rows(
  tibble(model = "DeepSD", error = df$error_deepsd),
  tibble(model = "BilinearBaseline", error = df$error_baseline)
)
hist_plot <- ggplot(err_long, aes(x = error, fill = model)) +
  geom_histogram(bins = 45, alpha = 0.55, position = "identity", color = "white") +
  labs(
    title = "Prediction Error Distributions",
    x = "Error (prediction - truth)",
    y = "Count"
  ) +
  theme_minimal(base_size = 12)

# Extreme RMSE comparison
if (nrow(extreme_df) > 0) {
  extreme_plot <- extreme_df %>%
    mutate(family = ifelse(grepl("^DeepSD", model), "DeepSD", "Baseline")) %>%
    ggplot(aes(x = factor(percentile), y = rmse, fill = family)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.65) +
    labs(
      title = "RMSE for Extreme Precipitation Thresholds",
      x = "Percentile threshold",
      y = "RMSE",
      fill = "Model"
    ) +
    theme_minimal(base_size = 12)
  ggsave(
    file.path(output_dir, "r_extreme_rmse.png"),
    extreme_plot,
    width = 7.5,
    height = 4.8,
    dpi = 130
  )
}

ggsave(
  file.path(output_dir, "r_scatter_pred_vs_true.png"),
  scatter_plot,
  width = 7,
  height = 5.5,
  dpi = 130
)
ggsave(
  file.path(output_dir, "r_error_histogram.png"),
  hist_plot,
  width = 7.5,
  height = 5,
  dpi = 130
)

cat("Saved:\n")
cat("- ", stats_path, "\n", sep = "")
cat("- ", t_test_path, "\n", sep = "")
cat("- ", extreme_path, "\n", sep = "")
cat("- ", file.path(output_dir, "r_scatter_pred_vs_true.png"), "\n", sep = "")
cat("- ", file.path(output_dir, "r_error_histogram.png"), "\n", sep = "")
if (nrow(extreme_df) > 0) {
  cat("- ", file.path(output_dir, "r_extreme_rmse.png"), "\n", sep = "")
}
