# Climate downscaling evaluation script in R
#
# This script demonstrates post-training diagnostics:
# 1) summary statistics (mean, median, sd)
# 2) paired t-test (predictions vs truth)
# 3) diagnostic visualizations
#
# Expected input CSV schema:
#   y_true, y_pred

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
  message("Creating a synthetic example dataset at outputs/predictions.csv")
  set.seed(42)
  y_true <- rnorm(8000, mean = 28, sd = 4)
  y_pred <- y_true + rnorm(8000, mean = 0.15, sd = 1.1)
  demo <- data.frame(y_true = y_true, y_pred = y_pred)
  write_csv(demo, "outputs/predictions.csv")
  input_path <- "outputs/predictions.csv"
}

df <- read_csv(input_path, show_col_types = FALSE) %>%
  mutate(error = y_pred - y_true)

required_cols <- c("y_true", "y_pred")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

# Descriptive statistics
stats <- df %>%
  summarise(
    n = n(),
    mean_true = mean(y_true),
    median_true = median(y_true),
    sd_true = sd(y_true),
    mean_pred = mean(y_pred),
    median_pred = median(y_pred),
    sd_pred = sd(y_pred),
    mean_error = mean(error),
    median_error = median(error),
    sd_error = sd(error),
    mae = mean(abs(error)),
    rmse = sqrt(mean(error^2))
  )

# Paired t-test for mean difference
tt <- t.test(df$y_pred, df$y_true, paired = TRUE)

stats_path <- file.path(output_dir, "r_statistics_summary.csv")
write_csv(stats, stats_path)

t_test_path <- file.path(output_dir, "r_paired_t_test.txt")
sink(t_test_path)
cat("Paired t-test: y_pred vs y_true\n")
print(tt)
sink()

# Visualization: scatter
scatter_plot <- ggplot(df, aes(x = y_true, y = y_pred)) +
  geom_point(alpha = 0.2, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs True Temperature",
    x = "True Temperature",
    y = "Predicted Temperature"
  ) +
  theme_minimal()

# Visualization: error distribution
hist_plot <- ggplot(df, aes(x = error)) +
  geom_histogram(bins = 40, fill = "darkorange", color = "white") +
  labs(
    title = "Prediction Error Distribution",
    x = "Error (y_pred - y_true)",
    y = "Count"
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "r_scatter_pred_vs_true.png"), scatter_plot, width = 7, height = 5, dpi = 120)
ggsave(file.path(output_dir, "r_error_histogram.png"), hist_plot, width = 7, height = 5, dpi = 120)

cat("Saved:\n")
cat("- ", stats_path, "\n", sep = "")
cat("- ", t_test_path, "\n", sep = "")
cat("- ", file.path(output_dir, "r_scatter_pred_vs_true.png"), "\n", sep = "")
cat("- ", file.path(output_dir, "r_error_histogram.png"), "\n", sep = "")
