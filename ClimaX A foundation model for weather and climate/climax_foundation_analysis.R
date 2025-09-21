# ClimaX Research Review — Reproducible Analysis (R)
#
# Educational companion to Nguyen et al. (ICML 2023). Generates synthetic
# forecast-skill, ClimateBench-style, downscaling, and scaling outputs that
# mirror the paper's evaluation narrative. Not a reproduction of official
# microsoft/ClimaX checkpoints or ERA5/CMIP6 downloads.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

set.seed(2023)

lead_hours <- c(6, 12, 24, 48, 72, 120, 168, 240, 336, 720)
variables <- c("t2m", "t850", "z500", "u10")
n_samples <- 250

ensure_dirs <- function() {
  dirs <- c("data/raw", "data/processed", "outputs", "assets")
  for (d in dirs) {
    dir.create(d, showWarnings = FALSE, recursive = TRUE)
  }
  file.create("data/raw/.gitkeep", showWarnings = FALSE)
  file.create("data/processed/.gitkeep", showWarnings = FALSE)
  file.create("outputs/.gitkeep", showWarnings = FALSE)
}

variable_difficulty <- function(variable) {
  switch(
    variable,
    "t2m" = 1.0,
    "t850" = 1.08,
    "z500" = 0.95,
    "u10" = 1.12,
    1.0
  )
}

model_scale <- function(model, lead_h) {
  if (model == "task_specific_dl") {
    1.05 + 0.0018 * lead_h
  } else if (model == "climax_like") {
    1.00 + 0.0011 * lead_h
  } else if (model == "nwp_like") {
    0.98 + 0.0016 * lead_h
  } else {
    stop(paste("Unknown model:", model))
  }
}

build_forecast_dataset <- function() {
  rows <- list()
  idx <- 1
  models <- c("task_specific_dl", "climax_like", "nwp_like")

  for (variable in variables) {
    for (lead_h in lead_hours) {
      for (model in models) {
        base <- (0.85 + 0.008 * lead_h) * variable_difficulty(variable)
        scale <- model_scale(model, lead_h)
        errors <- scale * base * rlnorm(n_samples, meanlog = 0.0, sdlog = 0.32)
        for (i in seq_len(n_samples)) {
          rows[[idx]] <- data.frame(
            sample_id = i - 1,
            variable = variable,
            lead_hours = lead_h,
            model = model,
            abs_error = errors[i],
            stringsAsFactors = FALSE
          )
          idx <- idx + 1
        }
      }
    }
  }
  bind_rows(rows)
}

summarize_forecast_skill <- function(df) {
  df %>%
    group_by(model, variable, lead_hours) %>%
    summarise(
      mae = mean(abs_error),
      median_ae = median(abs_error),
      std_ae = sd(abs_error),
      rmse = sqrt(mean(abs_error^2)),
      .groups = "drop"
    ) %>%
    arrange(variable, lead_hours, model)
}

paired_ttests_climax_vs_baseline <- function(df) {
  wide <- df %>%
    filter(model %in% c("task_specific_dl", "climax_like")) %>%
    select(sample_id, variable, lead_hours, model, abs_error) %>%
    tidyr::pivot_wider(names_from = model, values_from = abs_error)

  wide %>%
    group_by(variable, lead_hours) %>%
    summarise(
      t_stat = t.test(
        task_specific_dl,
        climax_like,
        paired = TRUE,
        alternative = "greater"
      )$statistic[[1]],
      p_value = t.test(
        task_specific_dl,
        climax_like,
        paired = TRUE,
        alternative = "greater"
      )$p.value,
      mean_diff_baseline_minus_climax = mean(task_specific_dl - climax_like),
      .groups = "drop"
    ) %>%
    arrange(variable, lead_hours)
}

build_climatebench_scores <- function() {
  set.seed(2030)
  models <- c("MLP", "CNN", "RandomForest", "ClimaX_like")
  targets <- c("tas", "pr", "diurnal_temperature_range")
  base <- c(MLP = 0.62, CNN = 0.71, RandomForest = 0.68, ClimaX_like = 0.78)

  expand.grid(target = targets, model = models, stringsAsFactors = FALSE) %>%
    mutate(
      score = pmin(pmax(rnorm(n(), mean = base[model], sd = 0.03), 0), 1)
    )
}

build_downscaling_metrics <- function() {
  set.seed(2034)
  models <- c("Bilinear", "CNN_baseline", "ClimaX_like")
  vars <- c("t2m", "t850", "z500")
  relative <- c(Bilinear = 1.25, CNN_baseline = 1.05, ClimaX_like = 0.92)

  expand.grid(variable = vars, model = models, stringsAsFactors = FALSE) %>%
    mutate(
      rmse = pmax(
        relative[model] * sapply(variable, variable_difficulty) * rnorm(n(), 1.0, 0.04),
        0.05
      )
    )
}

build_scaling_curve <- function() {
  set.seed(2036)
  params_millions <- c(5, 15, 50, 100, 200)
  mae_3day_t850 <- pmax(2.4 * (params_millions^-0.18) + rnorm(length(params_millions), 0, 0.03), 0.5)
  data.frame(
    params_millions = params_millions,
    mae_3day_t850 = mae_3day_t850,
    pretrain_datasets = 1:5
  )
}

ensure_dirs()

forecast_df <- build_forecast_dataset()
summary_df <- summarize_forecast_skill(forecast_df)
ttests <- paired_ttests_climax_vs_baseline(forecast_df)
climatebench <- build_climatebench_scores()
downscaling <- build_downscaling_metrics()
scaling <- build_scaling_curve()

write.csv(forecast_df, "data/processed/forecast_abs_errors_r.csv", row.names = FALSE)
write.csv(summary_df, "data/processed/forecast_skill_summary_r.csv", row.names = FALSE)
write.csv(ttests, "data/processed/forecast_paired_ttests_r.csv", row.names = FALSE)
write.csv(climatebench, "data/processed/climatebench_scores_r.csv", row.names = FALSE)
write.csv(downscaling, "data/processed/downscaling_rmse_r.csv", row.names = FALSE)
write.csv(scaling, "data/processed/scaling_curve_r.csv", row.names = FALSE)

write.csv(summary_df, "outputs/forecast_skill_summary_r.csv", row.names = FALSE)
write.csv(ttests, "outputs/forecast_paired_ttests_r.csv", row.names = FALSE)
write.csv(climatebench, "outputs/climatebench_scores_r.csv", row.names = FALSE)
write.csv(downscaling, "outputs/downscaling_rmse_r.csv", row.names = FALSE)
write.csv(scaling, "outputs/scaling_curve_r.csv", row.names = FALSE)

p_skill <- summary_df %>%
  filter(variable == "t2m") %>%
  ggplot(aes(x = lead_hours, y = mae, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Synthetic global forecast skill (T2m) — lower MAE is better",
    x = "Lead time (hours)",
    y = "Mean absolute error (a.u.)",
    color = "Model"
  ) +
  theme_minimal(base_size = 12)

ggsave("outputs/forecast_skill_t2m_r.png", p_skill, width = 10, height = 5, dpi = 160)
ggsave("assets/overview_r.png", p_skill, width = 10, height = 5, dpi = 150)

p_cb <- ggplot(climatebench, aes(x = target, y = score, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  labs(
    title = "Synthetic ClimateBench-style projection scores (higher is better)",
    x = "Target variable",
    y = "Score",
    fill = "Model"
  ) +
  ylim(0, 1.05) +
  theme_minimal(base_size = 12)

ggsave("outputs/climatebench_scores_r.png", p_cb, width = 9, height = 5, dpi = 160)

p_ds <- ggplot(downscaling, aes(x = variable, y = rmse, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  labs(
    title = "Synthetic downscaling RMSE (lower is better)",
    x = "Variable",
    y = "RMSE (a.u.)",
    fill = "Model"
  ) +
  theme_minimal(base_size = 12)

ggsave("outputs/downscaling_rmse_r.png", p_ds, width = 8, height = 5, dpi = 160)

p_sc <- ggplot(scaling, aes(x = params_millions, y = mae_3day_t850)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10() +
  labs(
    title = "Synthetic scaling: model size vs 3-day T850 MAE",
    x = "Parameters (millions, log scale)",
    y = "MAE (a.u.)"
  ) +
  theme_minimal(base_size = 12)

ggsave("outputs/scaling_curve_r.png", p_sc, width = 8, height = 5, dpi = 160)

long_lead <- summary_df %>%
  filter(variable == "t2m", lead_hours == max(lead_hours)) %>%
  arrange(mae)

cat("Wrote outputs to:", normalizePath("outputs"), "\n")
cat("Longest-lead T2m MAE ranking (synthetic):\n")
print(long_lead[, c("model", "mae", "rmse")])
sig_n <- sum(ttests$variable == "t2m" & ttests$p_value < 0.05)
cat(
  sprintf(
    "\nT2m leads with climax_like < task_specific_dl (p<0.05): %d/%d\n",
    sig_n,
    length(lead_hours)
  )
)
cat("\nNote: synthetic demo for education/review — not official ClimaX scores.\n")
