suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

set.seed(7)

lead_hours <- c(6, 12, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240)
n_samples <- 200

make_synthetic_errors <- function(n, lead_h, model) {
  base <- 1.0 + 0.010 * lead_h
  noise <- rlnorm(n, meanlog = 0.0, sdlog = 0.35)

  scale <- if (model == "baseline") {
    1.15
  } else if (model == "graphcast_like") {
    0.95
  } else {
    stop(paste("Unknown model:", model))
  }

  scale * base * noise
}

rows <- list()
idx <- 1
for (lead_h in lead_hours) {
  e_base <- make_synthetic_errors(n_samples, lead_h, "baseline")
  e_gc <- make_synthetic_errors(n_samples, lead_h, "graphcast_like")

  for (i in 1:n_samples) {
    rows[[idx]] <- data.frame(
      sample_id = i - 1,
      lead_hours = lead_h,
      baseline_abs_error = e_base[i],
      graphcast_like_abs_error = e_gc[i]
    )
    idx <- idx + 1
  }
}

df <- bind_rows(rows) %>%
  mutate(error_diff = baseline_abs_error - graphcast_like_abs_error)

summary <- df %>%
  group_by(lead_hours) %>%
  summarise(
    baseline_mae = mean(baseline_abs_error),
    graphcast_like_mae = mean(graphcast_like_abs_error),
    diff_mean = mean(error_diff),
    diff_sd = sd(error_diff),
    .groups = "drop"
  ) %>%
  arrange(lead_hours)

ttests <- df %>%
  group_by(lead_hours) %>%
  summarise(
    t_stat = t.test(baseline_abs_error, graphcast_like_abs_error, paired = TRUE, alternative = "greater")$statistic[[1]],
    p_value = t.test(baseline_abs_error, graphcast_like_abs_error, paired = TRUE, alternative = "greater")$p.value,
    .groups = "drop"
  ) %>%
  arrange(lead_hours)

outdir <- "outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

write.csv(df, file.path(outdir, "toy_skill_samples_r.csv"), row.names = FALSE)
write.csv(summary, file.path(outdir, "toy_skill_summary_r.csv"), row.names = FALSE)
write.csv(ttests, file.path(outdir, "toy_skill_ttests_r.csv"), row.names = FALSE)

p <- ggplot(summary, aes(x = lead_hours)) +
  geom_line(aes(y = baseline_mae, color = "Baseline"), linewidth = 1) +
  geom_point(aes(y = baseline_mae, color = "Baseline"), size = 2) +
  geom_line(aes(y = graphcast_like_mae, color = "GraphCast-like"), linewidth = 1) +
  geom_point(aes(y = graphcast_like_mae, color = "GraphCast-like"), size = 2) +
  labs(
    title = "Toy skill curve (lower MAE is better)",
    x = "Lead time (hours)",
    y = "Mean absolute error (a.u.)",
    color = "Model"
  ) +
  theme_minimal(base_size = 12)

ggsave(filename = file.path(outdir, "toy_skill_curve_r.png"), plot = p, width = 10, height = 5, dpi = 160)

cat("Wrote outputs to:", normalizePath(outdir), "\n")

