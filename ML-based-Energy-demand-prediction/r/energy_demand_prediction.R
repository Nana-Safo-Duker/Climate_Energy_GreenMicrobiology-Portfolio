suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(lubridate)
})

make_demo_data <- function(n_hours = 24 * 365, seed = 7) {
  set.seed(seed)
  ts <- seq.POSIXt(from = as.POSIXct("2022-01-01 00:00:00", tz = "UTC"),
                  by = "hour",
                  length.out = n_hours)
  df <- tibble(timestamp = ts) %>%
    mutate(
      hour = hour(timestamp),
      dow = wday(timestamp, week_start = 1) - 1,
      month = month(timestamp),
      dayofyear = yday(timestamp)
    )

  temp_daily <- 8 * sin(2 * pi * (df$hour / 24.0) - 1.2)
  temp_season <- 12 * sin(2 * pi * (df$dayofyear / 365.25) - 0.5)
  temperature_c <- 18 + temp_daily + temp_season + rnorm(nrow(df), 0, 1.0)

  is_weekend <- ifelse(df$dow >= 5, 1, 0)
  base <- 1200 + 80 * cos(2 * pi * (df$hour / 24.0))
  heating <- pmax(18 - temperature_c, 0) * 45
  cooling <- pmax(temperature_c - 22, 0) * 55
  weekend_drop <- is_weekend * 120
  noise <- rnorm(nrow(df), 0, 35)

  demand_mw <- base + heating + cooling - weekend_drop + noise

  df <- df %>%
    mutate(
      temperature_c = temperature_c,
      demand_mw = demand_mw,
      lag_1 = dplyr::lag(demand_mw, 1),
      lag_24 = dplyr::lag(demand_mw, 24),
      lag_168 = dplyr::lag(demand_mw, 168),
      roll_mean_24 = zoo::rollmean(dplyr::lag(demand_mw, 1), k = 24, fill = NA, align = "right"),
      roll_sd_24 = zoo::rollapply(dplyr::lag(demand_mw, 1), width = 24, FUN = sd, fill = NA, align = "right")
    ) %>%
    tidyr::drop_na()

  df
}

time_split <- function(df, train_frac = 0.8) {
  n <- nrow(df)
  split <- floor(n * train_frac)
  list(train = df[1:split, ], test = df[(split + 1):n, ])
}

mae <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat) ^ 2))

run_demo <- function() {
  if (!requireNamespace("zoo", quietly = TRUE) ||
      !requireNamespace("tidyr", quietly = TRUE)) {
    stop("Please install packages: zoo, tidyr")
  }

  df <- make_demo_data()
  sp <- time_split(df, 0.8)
  train <- sp$train
  test <- sp$test

  # Simple linear baseline with engineered features
  # Use numeric seasonal encodings to avoid "new factor levels" issues
  # when training on early months and testing on later months.
  train <- train %>%
    mutate(
      dow_sin = sin(2 * pi * dow / 7),
      dow_cos = cos(2 * pi * dow / 7),
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12)
    )

  test <- test %>%
    mutate(
      dow_sin = sin(2 * pi * dow / 7),
      dow_cos = cos(2 * pi * dow / 7),
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12)
    )

  fit <- lm(demand_mw ~ hour + temperature_c + dow_sin + dow_cos + month_sin + month_cos +
              lag_1 + lag_24 + lag_168 + roll_mean_24 + roll_sd_24,
            data = train)
  pred <- predict(fit, newdata = test)

  cat(sprintf("Rows: train=%d test=%d\n", nrow(train), nrow(test)))
  cat(sprintf("MAE:  %.2f MW\n", mae(test$demand_mw, pred)))
  cat(sprintf("RMSE: %.2f MW\n", rmse(test$demand_mw, pred)))

  p <- tibble(
    timestamp = test$timestamp,
    actual = test$demand_mw,
    predicted = as.numeric(pred)
  ) %>%
    slice(1:7*24) %>%
    tidyr::pivot_longer(cols = c(actual, predicted), names_to = "series", values_to = "demand_mw") %>%
    ggplot(aes(x = timestamp, y = demand_mw, color = series)) +
    geom_line(linewidth = 0.7) +
    labs(title = "Energy demand: actual vs predicted (first test week)",
         x = NULL, y = "Demand (MW)") +
    theme_minimal(base_size = 12)

  print(p)
}

run_demo()
