# Continuous Ensemble Weather Forecasting — R diagnostics
# Educational companion to Andrae et al. (2024), arXiv:2410.05431
#
# This script mirrors the Python demo's probabilistic evaluation:
# - synthetic lead-time ensemble trajectories
# - RMSE of ensemble mean, CRPS (Gaussian), Spread/Skill Ratio
# - temporal-difference continuity diagnostic
#
# Usage (from project root):
#   Rscript r/continuous_ensemble_forecasting.R

suppressPackageStartupMessages({
  library(stats)
  library(utils)
  library(graphics)
  library(grDevices)
})

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("assets", showWarnings = FALSE, recursive = TRUE)
set.seed(42)

# ---- Configuration ----
n_lat <- 16L
n_lon <- 32L
n_ens <- 30L
total_hours <- 240L
ar_step <- 24L
interp <- 6L
rho <- log(10)
noise_scale <- 2.5

lat <- seq(-75, 75, length.out = n_lat)
lon <- seq(-180, 179.999, length.out = n_lon)
lon_g <- matrix(rep(lon, each = n_lat), nrow = n_lat)
lat_g <- matrix(rep(lat, times = n_lon), nrow = n_lat)

true_field <- function(t_hours, add_noise = FALSE) {
  seasonal <- 12 * sin(2 * pi * (t_hours %% 8760) / 8760)
  wave <- 6 * sin(2 * pi * (lon_g / 360) - 2 * pi * t_hours / 120)
  meridional <- 8 * cos(lat_g * pi / 180)
  field <- 273.15 - 20 + seasonal + wave + meridional
  if (add_noise) {
    field <- field + matrix(rnorm(n_lat * n_lon, 0, 0.4), nrow = n_lat)
  }
  field
}

score_conditioned_mean <- function(init0, init_m, lead_h) {
  tendency <- init0 - init_m
  persistence <- init0 + tendency * (lead_h / 6) * 0.15
  climate <- true_field(lead_h, add_noise = FALSE)
  w <- exp(-lead_h / 72)
  w * persistence + (1 - w) * climate
}

ou_noise_member <- function(n_times, rho) {
  z <- array(0, dim = c(n_times, n_lat, n_lon))
  z[1, , ] <- matrix(rnorm(n_lat * n_lon), nrow = n_lat)
  if (n_times >= 2) {
    for (i in 2:n_times) {
      alpha <- exp(-rho)
      innov <- matrix(rnorm(n_lat * n_lon), nrow = n_lat)
      z[i, , ] <- alpha * z[i - 1, , ] + sqrt(max(1 - alpha^2, 0)) * innov
    }
  }
  z
}

continuous_ensemble <- function(init0, init_m, lead_hours, n_ens, rho, noise_scale) {
  n_leads <- length(lead_hours)
  out <- array(0, dim = c(n_ens, n_leads, n_lat, n_lon))
  for (k in seq_len(n_ens)) {
    noise <- ou_noise_member(n_leads, rho)
    for (j in seq_along(lead_hours)) {
      lead <- lead_hours[j]
      mu <- score_conditioned_mean(init0, init_m, lead)
      scale <- noise_scale * (0.7 + 0.3 * (1 - exp(-lead / 48)))
      out[k, j, , ] <- mu + scale * noise[j, , ]
    }
  }
  out
}

arci_forecast <- function(init0, init_m, total_hours, ar_step, interp, n_ens, rho, noise_scale) {
  leads_block <- seq(interp, ar_step, by = interp)
  n_blocks <- ceiling(total_hours / ar_step)
  members_list <- list()
  leads <- c()
  cur0 <- init0
  cur_m <- init_m
  t0 <- 0
  for (b in seq_len(n_blocks)) {
    block <- continuous_ensemble(cur0, cur_m, leads_block, n_ens, rho, noise_scale)
    for (j in seq_along(leads_block)) {
      members_list[[length(members_list) + 1]] <- block[, j, , ]
      leads <- c(leads, t0 + leads_block[j])
    }
    # ensemble-mean advance at AR anchor
    cur_m <- cur0
    cur0 <- apply(block[, length(leads_block), , ], c(2, 3), mean)
    t0 <- t0 + ar_step
  }
  n_times <- length(leads)
  ens <- array(0, dim = c(n_ens, n_times, n_lat, n_lon))
  for (t in seq_len(n_times)) {
    ens[, t, , ] <- members_list[[t]]
  }
  list(ens = ens, leads = leads)
}

rmse <- function(a, b) sqrt(mean((a - b)^2))

crps_gaussian <- function(obs, mu, sigma) {
  sigma <- pmax(sigma, 1e-6)
  z <- (obs - mu) / sigma
  scores <- sigma * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi))
  mean(scores)
}

ssr <- function(ens_slice, truth) {
  # ens_slice: n_ens x lat x lon
  ens_mean <- apply(ens_slice, c(2, 3), mean)
  spread <- sqrt(mean(apply(ens_slice, c(2, 3), var)))
  skill <- rmse(ens_mean, truth)
  spread / max(skill, 1e-8)
}

# ---- Run ARCI demo ----
init0 <- true_field(0, add_noise = TRUE)
init_m <- true_field(-6, add_noise = TRUE)
res <- arci_forecast(init0, init_m, total_hours, ar_step, interp, n_ens, rho, noise_scale)
ens <- res$ens
leads <- res$leads

metrics <- data.frame(
  lead_hours = leads,
  rmse = NA_real_,
  crps = NA_real_,
  ssr = NA_real_
)

for (j in seq_along(leads)) {
  truth <- true_field(leads[j], add_noise = TRUE)
  members <- ens[, j, , ]
  mu <- apply(members, c(2, 3), mean)
  sigma <- apply(members, c(2, 3), sd)
  metrics$rmse[j] <- rmse(mu, truth)
  metrics$crps[j] <- crps_gaussian(truth, mu, sigma)
  metrics$ssr[j] <- ssr(members, truth)
}

write.csv(metrics, "outputs/ensemble_metrics_r.csv", row.names = FALSE)

# Continuity diagnostic at grid center
i <- ceiling(n_lat / 2)
j <- ceiling(n_lon / 2)
ens_ts <- ens[, , i, j]
truth_ts <- sapply(leads, function(h) true_field(h, add_noise = FALSE)[i, j])
temporal_diff <- mean(abs(diff(colMeans(ens_ts))))

summary_txt <- sprintf(
  paste0(
    "R continuous ensemble demo\n",
    "n_times=%d  mean_RMSE=%.4f  mean_CRPS=%.4f  mean_SSR=%.4f\n",
    "mean |Δ| (ens mean, center)=%.4f\n"
  ),
  length(leads),
  mean(metrics$rmse),
  mean(metrics$crps),
  mean(metrics$ssr),
  temporal_diff
)
cat(summary_txt)
writeLines(summary_txt, "outputs/summary_r.txt")

# ---- Figures ----
png("outputs/overview_r.png", width = 1100, height = 800, res = 140)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(metrics$lead_hours, metrics$rmse, type = "l", col = "#1f4e79", lwd = 2,
     xlab = "Lead time (h)", ylab = "Score", main = "RMSE and CRPS vs lead time")
lines(metrics$lead_hours, metrics$crps, col = "#c45c26", lwd = 2)
legend("topleft", legend = c("RMSE", "CRPS"), col = c("#1f4e79", "#c45c26"), lwd = 2, bty = "n")

plot(metrics$lead_hours, metrics$ssr, type = "l", col = "#2a6f4e", lwd = 2,
     xlab = "Lead time (h)", ylab = "SSR", main = "Spread/Skill Ratio")
abline(h = 1, lty = 2, col = "gray40")

matplot(leads, t(ens_ts[1:min(8, n_ens), ]), type = "l", lty = 1, col = rgb(0.48, 0.64, 0.77, 0.45),
        xlab = "Lead time (h)", ylab = "t850 proxy (K)",
        main = "Ensemble trajectories (grid center)")
lines(leads, colMeans(ens_ts), col = "#1f4e79", lwd = 2)
lines(leads, truth_ts, col = "#c45c26", lwd = 2)

final_mean <- apply(ens[, length(leads), , ], c(2, 3), mean)
final_truth <- true_field(leads[length(leads)], add_noise = FALSE)
image(t(final_mean - final_truth)[, n_lat:1], col = hcl.colors(40, "Blue-Red"),
      axes = FALSE, main = "Ens. mean − truth (final lead)")

dev.off()

file.copy("outputs/overview_r.png", "assets/overview_r.png", overwrite = TRUE)
cat("Wrote outputs/ensemble_metrics_r.csv, outputs/summary_r.txt, outputs/overview_r.png\n")
