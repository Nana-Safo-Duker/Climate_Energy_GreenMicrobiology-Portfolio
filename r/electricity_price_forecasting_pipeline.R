# Electricity Price Index Forecasting Pipeline (R)
# ------------------------------------------------
# This script mirrors the Python workflow with:
# - data loading and validation
# - time, lag, rolling feature engineering
# - model comparison using time-aware resampling
# - performance reporting

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(rsample)
  library(yardstick)
  library(slider)
  library(ranger)
})

cfg <- list(
  data_path = "data/raw/electricity_price_index.csv",
  timestamp_col = "timestamp",
  target_col = "price_index",
  output_dir = "outputs"
)

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

if (!file.exists(cfg$data_path)) {
  stop(
    paste0(
      "Input data not found at ", cfg$data_path,
      ". Create CSV with columns: timestamp, price_index"
    )
  )
}

df <- read_csv(cfg$data_path, show_col_types = FALSE) %>%
  mutate(
    !!cfg$timestamp_col := ymd_hms(.data[[cfg$timestamp_col]], quiet = TRUE)
  ) %>%
  drop_na(all_of(c(cfg$timestamp_col, cfg$target_col))) %>%
  arrange(.data[[cfg$timestamp_col]])

feature_df <- df %>%
  mutate(
    hour = hour(.data[[cfg$timestamp_col]]),
    dayofweek = wday(.data[[cfg$timestamp_col]], week_start = 1),
    month = month(.data[[cfg$timestamp_col]]),
    quarter = quarter(.data[[cfg$timestamp_col]]),
    is_weekend = if_else(dayofweek >= 6, 1, 0),
    lag_1 = lag(.data[[cfg$target_col]], 1),
    lag_2 = lag(.data[[cfg$target_col]], 2),
    lag_3 = lag(.data[[cfg$target_col]], 3),
    lag_6 = lag(.data[[cfg$target_col]], 6),
    lag_12 = lag(.data[[cfg$target_col]], 12),
    lag_24 = lag(.data[[cfg$target_col]], 24),
    roll_mean_3 = slide_dbl(.data[[cfg$target_col]], mean, .before = 2, .complete = TRUE),
    roll_mean_6 = slide_dbl(.data[[cfg$target_col]], mean, .before = 5, .complete = TRUE),
    roll_mean_12 = slide_dbl(.data[[cfg$target_col]], mean, .before = 11, .complete = TRUE),
    roll_std_3 = slide_dbl(.data[[cfg$target_col]], sd, .before = 2, .complete = TRUE),
    roll_std_6 = slide_dbl(.data[[cfg$target_col]], sd, .before = 5, .complete = TRUE),
    roll_std_12 = slide_dbl(.data[[cfg$target_col]], sd, .before = 11, .complete = TRUE)
  ) %>%
  drop_na()

initial_window <- floor(nrow(feature_df) * 0.6)
assess_window <- floor(nrow(feature_df) * 0.1)
skip_window <- floor(nrow(feature_df) * 0.05)

splits <- rolling_origin(
  feature_df,
  initial = initial_window,
  assess = assess_window,
  skip = skip_window,
  cumulative = TRUE
)

fit_and_score <- function(split_obj) {
  train_data <- analysis(split_obj)
  test_data <- assessment(split_obj)

  predictors <- setdiff(colnames(train_data), c(cfg$timestamp_col, cfg$target_col))
  formula_obj <- as.formula(
    paste(cfg$target_col, "~", paste(predictors, collapse = " + "))
  )

  lm_fit <- lm(formula_obj, data = train_data)
  rf_fit <- ranger(
    formula = formula_obj,
    data = train_data,
    num.trees = 300,
    mtry = max(2, floor(sqrt(length(predictors)))),
    importance = "impurity",
    seed = 42
  )

  pred_lm <- predict(lm_fit, newdata = test_data)
  pred_rf <- predict(rf_fit, data = test_data)$predictions

  truth <- test_data[[cfg$target_col]]

  tibble(
    model = c("LinearRegression", "RandomForest"),
    mae = c(mae_vec(truth, pred_lm), mae_vec(truth, pred_rf)),
    rmse = c(rmse_vec(truth, pred_lm), rmse_vec(truth, pred_rf)),
    rsq = c(rsq_vec(truth, pred_lm), rsq_vec(truth, pred_rf))
  )
}

metrics_by_split <- map_dfr(seq_along(splits$splits), function(i) {
  fit_and_score(splits$splits[[i]]) %>%
    mutate(split = i)
})

summary_table <- metrics_by_split %>%
  group_by(model) %>%
  summarise(
    mae_mean = mean(mae),
    mae_sd = sd(mae),
    rmse_mean = mean(rmse),
    rmse_sd = sd(rmse),
    rsq_mean = mean(rsq),
    rsq_sd = sd(rsq),
    .groups = "drop"
  ) %>%
  arrange(rmse_mean)

print(summary_table)

write_csv(metrics_by_split, file.path(cfg$output_dir, "cv_metrics_r.csv"))
write_csv(summary_table, file.path(cfg$output_dir, "cv_summary_r.csv"))

message("R pipeline completed. Outputs saved in ", cfg$output_dir)
