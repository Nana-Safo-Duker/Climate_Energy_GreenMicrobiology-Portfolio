suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

# This script is a lightweight companion to the toy DRL environment.
# It demonstrates how you might summarize/evaluate multiple experimental runs.

set.seed(42)

out_dir <- file.path("outputs", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Create a synthetic "results" dataset that mimics multiple runs / scenarios.
df <- tibble(
  run_id = 1:30,
  controller = rep(c("heuristic", "drl_policy"), each = 15),
  total_cost = c(rnorm(15, mean = 18.5, sd = 1.6), rnorm(15, mean = 16.2, sd = 1.4)),
  comfort_violation = c(rnorm(15, mean = 6.0, sd = 1.0), rnorm(15, mean = 4.8, sd = 1.0))
) %>%
  mutate(
    total_cost = pmax(total_cost, 0),
    comfort_violation = pmax(comfort_violation, 0)
  )

summary_tbl <- df %>%
  group_by(controller) %>%
  summarise(
    n = n(),
    mean_cost = mean(total_cost),
    sd_cost = sd(total_cost),
    mean_comfort = mean(comfort_violation),
    sd_comfort = sd(comfort_violation),
    .groups = "drop"
  )

write_csv(df, file.path(out_dir, "synthetic_run_results.csv"))
write_csv(summary_tbl, file.path(out_dir, "summary_statistics.csv"))

# Paired-style comparison isn't possible here because the synthetic data isn't paired by scenario,
# but we can still illustrate a basic hypothesis test for differences in means.
t_cost <- t.test(total_cost ~ controller, data = df)
t_comfort <- t.test(comfort_violation ~ controller, data = df)

sink(file.path(out_dir, "t_test_results.txt"))
cat("T-test: total_cost ~ controller\n")
print(t_cost)
cat("\nT-test: comfort_violation ~ controller\n")
print(t_comfort)
sink()

p1 <- ggplot(df, aes(x = controller, y = total_cost, fill = controller)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  labs(title = "Total cost by controller (synthetic)", x = NULL, y = "Total cost") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

p2 <- ggplot(df, aes(x = controller, y = comfort_violation, fill = controller)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  labs(title = "Comfort violation by controller (synthetic)", x = NULL, y = "Comfort violation") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "total_cost_boxplot.png"), p1, width = 7, height = 4, dpi = 160)
ggsave(file.path(out_dir, "comfort_violation_boxplot.png"), p2, width = 7, height = 4, dpi = 160)

print(summary_tbl)

