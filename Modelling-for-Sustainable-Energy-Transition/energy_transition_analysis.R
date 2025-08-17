# Energy Transition Research Review - Reproducible R Workflow
# This script mirrors the Python analysis using tidyverse-style operations.

suppressPackageStartupMessages({
  library(tidyverse)
})

ensure_directories <- function() {
  dirs <- c("data/raw", "data/processed", "outputs")
  for (d in dirs) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  file.create("data/raw/.gitkeep", showWarnings = FALSE)
  file.create("data/processed/.gitkeep", showWarnings = FALSE)
  file.create("outputs/.gitkeep", showWarnings = FALSE)
}

load_data <- function(path = "data/raw/transition_scenarios.csv") {
  if (file.exists(path)) {
    df <- readr::read_csv(path, show_col_types = FALSE)
  } else {
    set.seed(1010)
    scenarios <- c("baseline", "accelerated_transition")
    years <- 2025:2040
    df <- expand.grid(scenario = scenarios, year = years) %>%
      as_tibble() %>%
      mutate(
        renewable_base = 30 + (year - 2025) * if_else(scenario == "baseline", 1.2, 2.0),
        emissions_base = 500 - (year - 2025) * if_else(scenario == "baseline", 8, 15),
        cost_base = 220 + (year - 2025) * if_else(scenario == "baseline", 2, 1),
        renewable_share = pmin(pmax(rnorm(n(), renewable_base, 2.5), 0), 100),
        co2_emissions_mt = pmax(rnorm(n(), emissions_base, 10), 0),
        system_cost_billion_usd = pmax(rnorm(n(), cost_base, 4), 0)
      ) %>%
      select(scenario, year, renewable_share, co2_emissions_mt, system_cost_billion_usd)

    readr::write_csv(df, path)
  }
  df
}

descriptive_statistics <- function(df) {
  desc <- df %>%
    group_by(scenario) %>%
    summarise(
      renewable_mean = mean(renewable_share),
      renewable_median = median(renewable_share),
      renewable_sd = sd(renewable_share),
      emissions_mean = mean(co2_emissions_mt),
      emissions_median = median(co2_emissions_mt),
      emissions_sd = sd(co2_emissions_mt),
      cost_mean = mean(system_cost_billion_usd),
      cost_median = median(system_cost_billion_usd),
      cost_sd = sd(system_cost_billion_usd),
      .groups = "drop"
    )
  readr::write_csv(desc, "data/processed/descriptive_statistics_r.csv")
  desc
}

perform_ttests <- function(df) {
  t_renewable <- t.test(renewable_share ~ scenario, data = df)
  t_emissions <- t.test(co2_emissions_mt ~ scenario, data = df)
  t_cost <- t.test(system_cost_billion_usd ~ scenario, data = df)

  results <- tibble(
    variable = c("renewable_share", "co2_emissions_mt", "system_cost_billion_usd"),
    t_statistic = c(t_renewable$statistic, t_emissions$statistic, t_cost$statistic),
    p_value = c(t_renewable$p.value, t_emissions$p.value, t_cost$p.value)
  )
  readr::write_csv(results, "data/processed/ttest_results_r.csv")
  results
}

create_figures <- function(df) {
  p1 <- ggplot(df, aes(x = year, y = renewable_share, color = scenario)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    labs(
      title = "Renewable Share Over Time by Scenario",
      x = "Year",
      y = "Renewable Share (%)"
    ) +
    theme_minimal()
  ggsave("outputs/figure_renewable_share_trend_r.png", p1, width = 9, height = 5, dpi = 300)

  p2 <- ggplot(df, aes(x = year, y = co2_emissions_mt, color = scenario)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    labs(
      title = "CO2 Emissions Over Time by Scenario",
      x = "Year",
      y = "CO2 Emissions (Mt)"
    ) +
    theme_minimal()
  ggsave("outputs/figure_emissions_trend_r.png", p2, width = 9, height = 5, dpi = 300)

  p3 <- ggplot(df, aes(x = scenario, y = system_cost_billion_usd, fill = scenario)) +
    geom_boxplot(alpha = 0.8) +
    labs(
      title = "System Cost Distribution by Scenario",
      x = "Scenario",
      y = "System Cost (Billion USD)"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  ggsave("outputs/figure_cost_boxplot_r.png", p3, width = 8, height = 5, dpi = 300)
}

write_notes <- function(ttests) {
  lines <- c(
    "# R Analysis Notes",
    "",
    "This output summarizes Welch t-tests between baseline and accelerated scenarios.",
    "Interpret p-values below 0.05 as statistically notable differences (illustrative).",
    ""
  )
  for (i in seq_len(nrow(ttests))) {
    lines <- c(
      lines,
      sprintf(
        "- %s: t = %.3f, p = %.4f",
        ttests$variable[i], ttests$t_statistic[i], ttests$p_value[i]
      )
    )
  }
  writeLines(lines, "outputs/interpretation_notes_r.md")
}

main <- function() {
  ensure_directories()
  df <- load_data()
  descriptive_statistics(df)
  ttests <- perform_ttests(df)
  create_figures(df)
  write_notes(ttests)

  message("R analysis complete.")
}

main()
