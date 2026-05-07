suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(ggplot2)
  library(randomForest)
})

safe_mape <- function(y_true, y_pred) {
  denom <- pmax(abs(y_true), 1e-6)
  mean(abs((y_true - y_pred) / denom)) * 100.0
}

make_synthetic_hourly_dataset <- function(start = "2024-01-01", periods = 24 * 180, seed = 7) {
  set.seed(seed)
  dt <- seq.POSIXt(from = as.POSIXct(start, tz = "UTC"), by = "hour", length.out = periods)
  hour <- as.integer(format(dt, "%H"))
  dow <- as.integer(format(dt, "%u")) - 1
  doy <- as.integer(format(dt, "%j"))

  temp_c <- 12 +
    10 * sin(2 * pi * (doy / 365.25)) +
    5 * sin(2 * pi * (hour / 24.0)) +
    rnorm(periods, 0, 1.2)

  base <- 1200.0
  daily <- 180 * sin(2 * pi * (hour / 24.0 - 0.15))
  weekly <- ifelse(dow < 5, 90, -60)
  comfort <- 18.0
  temp_effect <- 15.0 * pmax(comfort - temp_c, 0) + 9.0 * pmax(temp_c - comfort, 0)
  noise <- rnorm(periods, 0, 35)

  load <- pmax(base + daily + weekly + temp_effect + noise, 50.0)

  tibble(
    datetime = dt,
    load = load,
    temp_c = temp_c
  )
}

add_time_features <- function(df) {
  df %>%
    mutate(
      datetime = as.POSIXct(datetime, tz = "UTC"),
      hour = hour(datetime),
      dayofweek = wday(datetime, week_start = 1) - 1,
      month = month(datetime),
      hour_sin = sin(2 * pi * hour / 24.0),
      hour_cos = cos(2 * pi * hour / 24.0),
      dow_sin = sin(2 * pi * dayofweek / 7.0),
      dow_cos = cos(2 * pi * dayofweek / 7.0)
    ) %>%
    arrange(datetime) %>%
    mutate(
      load_lag_1 = dplyr::lag(load, 1),
      load_lag_24 = dplyr::lag(load, 24),
      load_roll_mean_24 = zoo::rollmean(dplyr::lag(load, 1), k = 24, fill = NA, align = "right")
    ) %>%
    tidyr::drop_na()
}

time_series_cv_indices <- function(n, k = 5) {
  # Expanding window CV indices (simple implementation)
  fold_sizes <- floor((n) / (k + 1))
  indices <- vector("list", k)
  for (i in seq_len(k)) {
    train_end <- fold_sizes * i
    test_end <- fold_sizes * (i + 1)
    indices[[i]] <- list(
      train = seq_len(train_end),
      test = (train_end + 1):test_end
    )
  }
  indices
}

train_and_evaluate <- function(df, splits = 5, seed = 7) {
  set.seed(seed)
  df2 <- add_time_features(df)

  features <- c(
    "temp_c",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month",
    "load_lag_1", "load_lag_24", "load_roll_mean_24"
  )

  X <- df2 %>% select(all_of(features))
  y <- df2$load

  idx <- time_series_cv_indices(nrow(df2), k = splits)

  mae <- c()
  rmse <- c()
  mape <- c()

  for (fold in idx) {
    train_idx <- fold$train
    test_idx <- fold$test

    rf <- randomForest(
      x = X[train_idx, ],
      y = y[train_idx],
      ntree = 400,
      nodesize = 2
    )

    pred <- predict(rf, newdata = X[test_idx, ])
    y_test <- y[test_idx]

    mae <- c(mae, mean(abs(y_test - pred)))
    rmse <- c(rmse, sqrt(mean((y_test - pred)^2)))
    mape <- c(mape, safe_mape(y_test, pred))
  }

  tibble(
    model = "RandomForest",
    MAE = mean(mae),
    RMSE = mean(rmse),
    MAPE = mean(mape)
  )
}

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else NA

df <- if (is.na(csv_path)) {
  make_synthetic_hourly_dataset()
} else {
  read.csv(csv_path) %>%
    mutate(datetime = ymd_hms(datetime, tz = "UTC"))
}

# Optional deps used in feature generation
if (!requireNamespace("zoo", quietly = TRUE) || !requireNamespace("tidyr", quietly = TRUE)) {
  stop("Please install R packages: zoo, tidyr (plus dplyr, lubridate, ggplot2, randomForest).")
}

res <- train_and_evaluate(df, splits = 5, seed = 7)
print(res)

