suppressPackageStartupMessages({
  library(stats)
  library(utils)
  library(graphics)
  library(grDevices)
  library(rpart)
})

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
set.seed(42)

# ---- Synthetic "daily climate" table (replace with your real dataset) ----
n <- 5000
t <- 0:(n - 1)
day_of_year <- t %% 365
seasonal <- sin(2 * pi * day_of_year / 365)

temp_c <- 15 + 10 * seasonal + rnorm(n, 0, 2)
humidity_pct <- 60 - 15 * seasonal + rnorm(n, 0, 5)
pressure_hpa <- 1013 + rnorm(n, 0, 8)
wind_ms <- rgamma(n, shape = 2, scale = 1.5)

base <- 1.0 + 0.3 * (humidity_pct / 100) + 0.1 * wind_ms
spikes <- rbinom(n, 1, 0.06) * rlnorm(n, meanlog = 1.8, sdlog = 0.6)
target <- pmax(0, base + spikes + rnorm(n, 0, 0.2))

df <- data.frame(
  t = t,
  day_of_year = day_of_year,
  temp_c = temp_c,
  humidity_pct = humidity_pct,
  pressure_hpa = pressure_hpa,
  wind_ms = wind_ms,
  target = target
)

# ---- Extreme label (rare event classification) ----
extreme_quantile <- 0.95
threshold <- as.numeric(quantile(df$target, probs = extreme_quantile))
df$is_extreme <- as.integer(df$target >= threshold)

# ---- Train/test split ----
idx <- sample(seq_len(nrow(df)))
test_size <- 0.25
n_test <- floor(test_size * nrow(df))
test_idx <- idx[1:n_test]
train_idx <- idx[(n_test + 1):length(idx)]

train <- df[train_idx, ]
test <- df[test_idx, ]

features <- c("day_of_year", "temp_c", "humidity_pct", "pressure_hpa", "wind_ms")

# ---- Models: logistic regression + decision tree baseline ----
glm_fit <- glm(
  is_extreme ~ day_of_year + temp_c + humidity_pct + pressure_hpa + wind_ms,
  data = train,
  family = binomial()
)
glm_prob <- as.numeric(predict(glm_fit, newdata = test, type = "response"))
glm_pred <- as.integer(glm_prob >= 0.5)

rpart_fit <- rpart(
  is_extreme ~ day_of_year + temp_c + humidity_pct + pressure_hpa + wind_ms,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 20)
)
rpart_prob <- as.numeric(predict(rpart_fit, newdata = test, type = "prob")[, "1"])
rpart_pred <- as.integer(rpart_prob >= 0.5)

# ---- Metrics (event-focused) ----
confusion_metrics <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  c(precision = precision, recall = recall, f1 = f1, support_pos = sum(y_true == 1))
}

brier <- function(y_true, y_prob) mean((y_prob - y_true) ^ 2)

metrics <- rbind(
  data.frame(
    model = "logreg",
    extreme_quantile = extreme_quantile,
    threshold_target_value = threshold,
    t(confusion_metrics(test$is_extreme, glm_pred)),
    brier = brier(test$is_extreme, glm_prob)
  ),
  data.frame(
    model = "rpart",
    extreme_quantile = extreme_quantile,
    threshold_target_value = threshold,
    t(confusion_metrics(test$is_extreme, rpart_pred)),
    brier = brier(test$is_extreme, rpart_prob)
  )
)

write.csv(metrics, file = file.path("outputs", "metrics_r.csv"), row.names = FALSE)
print(metrics)

# ---- Figures ----
png(filename = file.path("outputs", "target_distribution_r.png"), width = 900, height = 500)
hist(df$target, breaks = 50, col = "gray80", border = "white",
     main = "Synthetic target distribution (with extreme threshold)",
     xlab = "Target")
abline(v = threshold, col = "red", lwd = 2, lty = 2)
dev.off()

