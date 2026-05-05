suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(slider)
  library(recipes)
  library(parsnip)
  library(workflows)
  library(rsample)
  library(yardstick)
})

root_dir <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), ".."), winslash = "/", mustWork = FALSE)
data_dir <- file.path(root_dir, "data")
reports_dir <- file.path(root_dir, "reports")
fig_dir <- file.path(reports_dir, "figures")

dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

dataset_path <- file.path(data_dir, "synthetic_renewable_timeseries.csv")

generate_synthetic_dataset <- function(n_days = 180, seed = 7) {
  set.seed(seed)
  idx <- seq.POSIXt(from = as.POSIXct("2025-01-01 00:00:00", tz = "UTC"),
                   by = "hour",
                   length.out = n_days * 24)
  n <- length(idx)
  hour <- hour(idx)
  dayofyear <- yday(idx)

  seasonal <- 0.65 + 0.35 * sin(2 * pi * (dayofyear / 365.25))
  diurnal <- pmax(sin(pi * (hour - 6) / 12), 0)
  cloud <- pmax(pmin(rnorm(n, 0, 0.18), 0.6), -0.6)
  irradiance <- pmax(seasonal * diurnal * (1 + cloud), 0)

  base_wind <- 7 + 2.0 * sin(2 * pi * (dayofyear / 365.25 + 0.25))
  ar <- rnorm(n, 0, 0.8)
  for (i in 2:n) ar[i] <- 0.85 * ar[i - 1] + ar[i]
  wind_speed <- pmax(base_wind + ar + rnorm(n, 0, 1.0), 0)

  temperature <- 12 + 10 * sin(2 * pi * (dayofyear / 365.25 - 0.1)) + rnorm(n, 0, 1.5)
  humidity <- pmin(pmax(55 + 20 * sin(2 * pi * (dayofyear / 365.25 + 0.05)) + rnorm(n, 0, 6), 10), 100)

  pv_power <- pmax(irradiance ^ 1.15 + rnorm(n, 0, 0.03), 0)
  wind_norm <- pmin(pmax((wind_speed - 3) / (12 - 3), 0), 1)
  wind_power <- pmin(pmax(wind_norm ^ 3 + rnorm(n, 0, 0.04), 0), 1)
  renewable_power <- pmin(pmax(0.55 * pv_power + 0.45 * wind_power + rnorm(n, 0, 0.03), 0), 1.2)

  df <- tibble(
    timestamp = idx,
    irradiance = irradiance,
    wind_speed = wind_speed,
    temperature_c = temperature,
    humidity_pct = humidity,
    hour = factor(hour),
    dayofyear = dayofyear,
    renewable_power = renewable_power
  )

  # add some missingness
  miss_cols <- c("irradiance", "wind_speed", "temperature_c", "humidity_pct")
  for (col in miss_cols) {
    mask <- runif(n) < 0.01
    df[[col]][mask] <- NA_real_
  }
  df
}

make_supervised <- function(df, horizon_hours = 1, n_lags = 24) {
  df <- df %>% arrange(timestamp)
  df <- df %>% mutate(target = lead(renewable_power, horizon_hours))
  for (lag_i in 1:n_lags) {
    nm <- paste0("lag_", lag_i)
    df[[nm]] <- dplyr::lag(df$renewable_power, lag_i)
  }
  df %>% drop_na()
}

rolling_backtest <- function(sup, test_size = 24 * 7, splits = 6, seed = 7) {
  set.seed(seed)
  n <- nrow(sup)
  fold_starts <- as.integer(seq(from = floor(n / (splits + 1)), to = n - test_size, length.out = splits))

  # recipe
  feature_cols <- setdiff(colnames(sup), c("target", "timestamp", "renewable_power"))
  rec <- recipe(target ~ ., data = sup %>% select(all_of(c("target", feature_cols)))) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

  # baseline model
  base_spec <- linear_reg(penalty = 1.0, mixture = 0) %>% set_engine("glmnet")
  base_wf <- workflow() %>% add_recipe(rec) %>% add_model(base_spec)

  # optimized: random forest with small grid
  rf_spec <- rand_forest(trees = tune(), min_n = tune()) %>%
    set_engine("ranger", num.threads = parallel::detectCores()) %>%
    set_mode("regression")
  rf_wf <- workflow() %>% add_recipe(rec) %>% add_model(rf_spec)

  grid <- crossing(trees = c(200L, 500L), min_n = c(2L, 5L, 10L))

  out <- vector("list", length(fold_starts))

  for (i in seq_along(fold_starts)) {
    start <- fold_starts[i]
    train <- sup[1:start, , drop = FALSE]
    test <- sup[(start + 1):(start + test_size), , drop = FALSE]

    # baseline fit/predict
    base_fit <- fit(base_wf, data = train %>% select(all_of(c("target", feature_cols))))
    base_pred <- predict(base_fit, new_data = test %>% select(all_of(feature_cols))) %>%
      bind_cols(test %>% select(timestamp, target))

    base_mae <- mae(base_pred, truth = target, estimate = .pred) %>% pull(.estimate)
    base_rmse <- rmse(base_pred, truth = target, estimate = .pred) %>% pull(.estimate)

    # "optimized" grid search using a time-series split on the training set
    # (kept simple so it runs anywhere; no heavy tuning frameworks required)
    inner_splits <- sliding_period(
      train %>% select(all_of(c("target", feature_cols))),
      index = train$timestamp,
      period = "month",
      lookback = 3,
      assess_stop = 1,
      step = 1,
      cumulative = TRUE
    )

    # If sliding_period produced too few splits, fall back to vfold
    if (length(inner_splits$splits) < 3) {
      inner_splits <- vfold_cv(train %>% select(all_of(c("target", feature_cols))), v = 5)
    }

    best <- NULL
    best_mae <- Inf

    for (g in seq_len(nrow(grid))) {
      spec_g <- finalize_model(rf_spec, grid[g, ])
      wf_g <- rf_wf %>% update_model(spec_g)

      # estimate MAE on inner splits
      fold_mae <- c()
      for (s in seq_along(inner_splits$splits)) {
        tr <- analysis(inner_splits$splits[[s]])
        va <- assessment(inner_splits$splits[[s]])
        fit_g <- fit(wf_g, data = tr)
        pred_g <- predict(fit_g, new_data = va %>% select(-target)) %>% bind_cols(va %>% select(target))
        fold_mae <- c(fold_mae, mae(pred_g, truth = target, estimate = .pred) %>% pull(.estimate))
      }
      avg_mae <- mean(fold_mae)
      if (avg_mae < best_mae) {
        best_mae <- avg_mae
        best <- grid[g, ]
      }
    }

    best_spec <- finalize_model(rf_spec, best)
    best_wf <- rf_wf %>% update_model(best_spec)
    best_fit <- fit(best_wf, data = train %>% select(all_of(c("target", feature_cols))))
    opt_pred <- predict(best_fit, new_data = test %>% select(all_of(feature_cols))) %>%
      bind_cols(test %>% select(timestamp, target))

    opt_mae <- mae(opt_pred, truth = target, estimate = .pred) %>% pull(.estimate)
    opt_rmse <- rmse(opt_pred, truth = target, estimate = .pred) %>% pull(.estimate)

    out[[i]] <- tibble(
      fold = i,
      test_start = min(test$timestamp),
      test_end = max(test$timestamp),
      baseline_mae = base_mae,
      baseline_rmse = base_rmse,
      optimized_mae = opt_mae,
      optimized_rmse = opt_rmse,
      best_params = paste0("trees=", best$trees, ", min_n=", best$min_n)
    )
  }

  bind_rows(out)
}

if (file.exists(dataset_path)) {
  df <- readr::read_csv(dataset_path, show_col_types = FALSE) %>%
    mutate(timestamp = ymd_hms(timestamp, tz = "UTC"),
           hour = factor(hour))
} else {
  df <- generate_synthetic_dataset()
  readr::write_csv(df, dataset_path)
}

sup <- make_supervised(df, horizon_hours = 1, n_lags = 24)
metrics <- rolling_backtest(sup, test_size = 24 * 7, splits = 6)

metrics_path <- file.path(reports_dir, "rolling_backtest_metrics_R.csv")
readr::write_csv(metrics, metrics_path)

plot_df <- metrics %>%
  select(fold, test_start, baseline_mae, optimized_mae) %>%
  pivot_longer(cols = c(baseline_mae, optimized_mae), names_to = "model", values_to = "mae") %>%
  mutate(model = recode(model,
                        baseline_mae = "Baseline (Elastic Net)",
                        optimized_mae = "Optimized (RF grid)"))

p <- ggplot(plot_df, aes(x = test_start, y = mae, color = model)) +
  geom_line() +
  geom_point() +
  theme_minimal(base_size = 12) +
  labs(title = "Rolling backtest MAE (lower is better)", x = "Test window start", y = "MAE", color = "Model")

ggsave(filename = file.path(fig_dir, "rolling_backtest_mae_R.png"), plot = p, width = 10, height = 4.8, dpi = 200)

message("Saved:")
message(paste0("- Dataset: ", dataset_path))
message(paste0("- Metrics: ", metrics_path))
message(paste0("- Figure:  ", file.path(fig_dir, "rolling_backtest_mae_R.png")))

