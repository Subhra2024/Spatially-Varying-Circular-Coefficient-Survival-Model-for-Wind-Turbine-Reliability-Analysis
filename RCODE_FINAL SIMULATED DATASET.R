# =============================================================================
# SVCC FRAMEWORK: MASTER ANALYSIS SCRIPT (FULLY SIMULATED VERSION)
# Includes All 10 Tables and 11 Advanced Diagnostic Plots
# Aligned with SVCC Methodology: Circular Continuity (Sec 3.1), 
# Spatial Non-Stationarity (Sec 3.2), Orthogonalization (Sec 3.3, Eq 5), 
# Penalized Likelihood (Sec 3.4, Eq 6), and IMRL (Sec 3.5, Eq 7).
# =============================================================================

# --- 1. Required Libraries ---
if (!require("survival")) install.packages("survival")
if (!require("survminer")) install.packages("survminer")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("pec")) install.packages("pec")
if (!require("car")) install.packages("car")
if (!require("viridis")) install.packages("viridis")
if (!require("reshape2")) install.packages("reshape2")

library(survival)
library(survminer)
library(ggplot2)
library(pec) 
library(car)
library(viridis)
library(reshape2)

# --- 2. Setup ---
output_dir <- "SVCC_SimulatedData_Results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
set.seed(42)
n_samples <- 1000 # Sample size for the simulation

# =============================================================================
# 3. DATA SIMULATION (METHODOLOGY SEC 3.1 & 3.2)
# =============================================================================
cat("Step 1: Simulating Turbine Dataset based on SVCC Framework...\n")

# Create a data frame with synthetic spatial and operational features
sim_df <- data.frame(
  id = 1:n_samples,
  # Simulate raw spatial coordinates (e.g., longitude and latitude)
  xlong = runif(n_samples, -85.5, -84.0),
  ylat = runif(n_samples, 42.0, 43.5)
)

# Normalize Spatial Coordinates (s_i in Eq 2)
sim_df$x <- (sim_df$xlong - min(sim_df$xlong)) / (max(sim_df$xlong) - min(sim_df$xlong))
sim_df$y <- (sim_df$ylat - min(sim_df$ylat)) / (max(sim_df$ylat) - min(sim_df$ylat))

# Circular Basis Mapping (Sec 3.1, Eq 1)
sim_df$wind_speed <- rnorm(n_samples, mean = 12, sd = 3)
sim_df$wd_rad <- runif(n_samples, 0, 2 * pi)
sim_df$sin_wd <- sin(sim_df$wd_rad)
sim_df$cos_wd <- cos(sim_df$wd_rad)

# Latent Risk Surface (Sec 3.2, Eq 3)
# Interaction weights emphasize localized risks based on coordinates and direction
linear_predictor <- 0.6 * sim_df$wind_speed + 
  0.4 * sim_df$sin_wd + 
  22.5 * (sim_df$x * sim_df$sin_wd) + 
  24.2 * (sim_df$y * sim_df$cos_wd)

# Survival Time Generation (Weibull-based survival simulation)
u <- runif(n_samples)
sim_df$time <- (-log(u) / (0.005 * exp(linear_predictor)))^(1/1.8)
sim_df$event <- rbinom(n_samples, 1, 0.90)

# =============================================================================
# 4. ORTHOGONALIZATION STRATEGY (SEC 3.3, EQ 5)
# =============================================================================
cat("Step 2: Performing Residual-Based Orthogonalization...\n")

# 1. Scale primary variables
sim_df$wind_speed_s <- as.numeric(scale(sim_df$wind_speed))
sim_df$sin_wd_s     <- as.numeric(scale(sim_df$sin_wd))
sim_df$cos_wd_s     <- as.numeric(scale(sim_df$cos_wd))

# 2. Advanced Orthogonalization (Eq 5)
# Removes linear signal of x and sin_wd from the interaction term
raw_x_sin <- sim_df$x * sim_df$sin_wd
raw_y_cos <- sim_df$y * sim_df$cos_wd

sim_df$x_sin <- as.numeric(scale(residuals(lm(raw_x_sin ~ x + sin_wd, data = sim_df))))
sim_df$y_cos <- as.numeric(scale(residuals(lm(raw_y_cos ~ y + cos_wd, data = sim_df))))

# Sensitivity Basis (Quadratic expansion for methodology validation)
sim_df$x_sin_quad <- as.numeric(scale(residuals(lm(I(sim_df$x^2 * sim_df$sin_wd) ~ x + sin_wd, data = sim_df))))
sim_df$y_cos_quad <- as.numeric(scale(residuals(lm(I(sim_df$y^2 * sim_df$cos_wd) ~ y + cos_wd, data = sim_df))))

# =============================================================================
# 5. MODEL ESTIMATION (SEC 3.4, EQ 6)
# =============================================================================
cat("Step 3: Fitting Models with Ridge Penalties (PPLL)...\n")

# Baseline model (No spatial interaction)
m_base <- coxph(Surv(time, event) ~ wind_speed_s + sin_wd_s + cos_wd_s, 
                data = sim_df, x = TRUE, y = TRUE)

# Proposed SVCC Model (Equation 6: Penalized Partial Likelihood)
m_svc  <- coxph(Surv(time, event) ~ wind_speed_s + sin_wd_s + cos_wd_s + 
                  ridge(x_sin, y_cos, theta = 0.5), 
                data = sim_df, x = TRUE, y = TRUE)

# Alternative Basis for Sensitivity Analysis
m_alt  <- coxph(Surv(time, event) ~ wind_speed_s + sin_wd_s + cos_wd_s + 
                  ridge(x_sin_quad, y_cos_quad, theta = 0.5), 
                data = sim_df, x = TRUE, y = TRUE)

# =============================================================================
# 6. GENERATE ALL 10 DIAGNOSTIC TABLES
# =============================================================================
cat("Step 4: Compiling 10 Diagnostic Tables...\n")

# --- Expected Failure Time (IMRL, Sec 3.5, Eq 7) ---
eval_times <- sort(unique(sim_df$time[sim_df$event == 1]))
eval_times <- eval_times[eval_times > quantile(eval_times, 0.1) & eval_times < quantile(eval_times, 0.8)]
eval_times <- seq(min(eval_times), max(eval_times), length.out = 20)

pred_expected_time <- function(model, newdata) {
  sf <- survfit(model, newdata = newdata)
  if (is.null(dim(sf$surv))) {
    return(sum(diff(c(0, sf$time)) * sf$surv))
  } else {
    expected_vals <- apply(sf$surv, 2, function(s) {
      sum(diff(c(0, sf$time)) * s)
    })
    return(expected_vals)
  }
}

# --- TABLE 1: Performance (MAE and IBS) ---
ibs_res <- tryCatch({
  pec(list(Baseline = m_base, Proposed_SVCC = m_svc), 
      formula = Surv(time, event) ~ 1, data = sim_df, 
      times = eval_times, exact = FALSE, cens.model = "marginal")
}, error = function(e) return(NULL))

pred_time_base <- pred_expected_time(m_base, sim_df)
pred_time_svc  <- pred_expected_time(m_svc, sim_df)

ibs_base_val <- if(!is.null(ibs_res$AppErr$Baseline)) tail(ibs_res$AppErr$Baseline, 1) else NA
ibs_svc_val  <- if(!is.null(ibs_res$AppErr$Proposed_SVCC)) tail(ibs_res$AppErr$Proposed_SVCC, 1) else NA

table1 <- data.frame(
  Metric = c("Mean Absolute Error (Time Scale)", "Integrated Brier Score (IBS)"),
  Baseline = c(mean(abs(sim_df$time - pred_time_base)), ibs_base_val),
  Proposed_SVCC = c(mean(abs(sim_df$time - pred_time_svc)), ibs_svc_val)
)
write.csv(table1, file.path(output_dir, "table1_performance.csv"), row.names = FALSE)

# --- TABLE 2: Advanced Multicollinearity ---
vif_mod <- lm(time ~ wind_speed_s + sin_wd_s + cos_wd_s + x_sin + y_cos, data = sim_df)
X_mat <- model.matrix(vif_mod)[,-1]
X_scaled <- scale(X_mat)
svd_d <- svd(X_scaled)$d
cond_indices <- max(svd_d) / svd_d

table2 <- data.frame(
  Predictor = names(vif(vif_mod)),
  VIF = as.numeric(vif(vif_mod)),
  Condition_Index = cond_indices[1:5],
  Corr_with_Target = as.numeric(cor(sim_df$time, X_mat))
)
write.csv(table2, file.path(output_dir, "table2_collinearity_diagnostics.csv"), row.names = FALSE)

# --- TABLE 3: Descriptive Stats ---
desc_stats <- as.data.frame(apply(sim_df[, c("time", "wind_speed", "x", "y")], 2, summary))
write.csv(desc_stats, file.path(output_dir, "table3_descriptive.csv"))

# --- TABLE 4 & 5: Model Coefficients ---
write.csv(as.data.frame(summary(m_base)$coefficients), file.path(output_dir, "table4_baseline_coefs.csv"))
write.csv(as.data.frame(summary(m_svc)$coefficients), file.path(output_dir, "table5_svcc_coefs.csv"))

# --- TABLE 6: Selection Metrics ---
table6 <- data.frame(
  Model = c("Base", "SVCC"), 
  AIC = c(AIC(m_base), AIC(m_svc)), 
  BIC = c(BIC(m_base), BIC(m_svc)),
  LogLik = c(as.numeric(logLik(m_base)), as.numeric(logLik(m_svc)))
)
write.csv(table6, file.path(output_dir, "table6_selection.csv"), row.names = FALSE)

# --- TABLE 7: Sensitivity Analysis (Quadratic Basis) ---
table7 <- data.frame(
  Scenario = c("Linear Basis", "Quadratic Basis"), 
  AIC = c(AIC(m_svc), AIC(m_alt)),
  BIC = c(BIC(m_svc), BIC(m_alt)),
  LogLik = c(as.numeric(logLik(m_svc)), as.numeric(logLik(m_alt)))
)
write.csv(table7, file.path(output_dir, "table7_sensitivity.csv"), row.names = FALSE)

# --- TABLE 8: Bootstrap Stability ---
cat("Running Bootstrapped Stability (N=1000 for simulation speed)...")
boot_ibs <- replicate(1000, { 
  idx <- sample(1:nrow(sim_df), replace = TRUE)
  b_df <- sim_df[idx, ]
  tryCatch({
    m_b <- coxph(Surv(time, event) ~ wind_speed_s + sin_wd_s + cos_wd_s + ridge(x_sin, y_cos, theta = 0.5), 
                 data = b_df, x = TRUE, y = TRUE)
    p_obj <- pec(list(S = m_b), formula = Surv(time, event) ~ 1, data = b_df, times = eval_times, verbose = FALSE, cens.model = "marginal")
    if(!is.null(p_obj$AppErr$S)) tail(p_obj$AppErr$S, 1) else NA
  }, error = function(e) return(NA))
})
boot_ibs <- boot_ibs[!is.na(boot_ibs)]
write.csv(data.frame(Metric = "Bootstrapped IBS", Mean = mean(boot_ibs), SD = sd(boot_ibs)), 
          file.path(output_dir, "table8_bootstrap.csv"), row.names = FALSE)

# --- TABLE 9: Concordance/Reliability ---
table9 <- data.frame(
  Metric = "Concordance (C-Index)", 
  Base = summary(m_base)$concordance[1], 
  SVCC = summary(m_svc)$concordance[1]
)
write.csv(table9, file.path(output_dir, "table9_reliability.csv"), row.names = FALSE)

# --- TABLE 10: Methodology Summary ---
table10 <- data.frame(
  Feature = c("AIC", "BIC", "LogLik", "C-Index"),
  Base = c(AIC(m_base), BIC(m_base), as.numeric(logLik(m_base)), summary(m_base)$concordance[1]),
  SVCC = c(AIC(m_svc), BIC(m_svc), as.numeric(logLik(m_svc)), summary(m_svc)$concordance[1])
)
write.csv(table10, file.path(output_dir, "table10_methodology_summary.csv"), row.names = FALSE)

# =============================================================================
# 7. GENERATE 11 ADVANCED PLOTS
# =============================================================================
cat("\nStep 5: Generating Diagnostic Visualizations...\n")

# P1: Spatial Failure Surface
p1 <- ggplot(sim_df, aes(x = x, y = y, fill = time)) + geom_point(size = 3, shape = 21, color = "black") + 
  scale_fill_viridis_c(option = "C") + labs(title = "Simulated Failure Time Risk Surface", fill = "Time") + theme_minimal()
ggsave(file.path(output_dir, "plot1_spatial.png"), p1)

# P2: Wind Speed Profile
p2 <- ggplot(sim_df, aes(x = wind_speed)) + geom_histogram(fill = "steelblue", color = "white", bins = 20) + 
  labs(title = "Simulated Wind Speed Profile", x = "m/s") + theme_minimal()
ggsave(file.path(output_dir, "plot2_wind.png"), p2)

# P3: Circular Continuity
sim_df$wd_deg <- (atan2(sim_df$sin_wd, sim_df$cos_wd) * 180 / pi) %% 360
p3 <- ggplot(sim_df, aes(x = wd_deg)) + geom_histogram(fill = "red", color = "black", bins = 16) + coord_polar() + 
  labs(title = "Simulated Directional Density") + theme_minimal()
ggsave(file.path(output_dir, "plot3_rose.png"), p3)

# P4: Kaplan-Meier
png(file.path(output_dir, "plot4_km.png"))
plot(survfit(Surv(time, event) ~ 1, data = sim_df), main = "Simulated Fleet Survival Function", col = "blue")
dev.off()

# P5: Cumulative Hazard
p5 <- ggplot(basehaz(m_svc), aes(x = time, y = hazard)) + geom_step(color = "darkgreen") + 
  labs(title = "Net Cumulative Hazard (Simulated)") + theme_minimal()
ggsave(file.path(output_dir, "plot5_hazard.png"), p5)

# P6: Coef Comparison
df_comb <- rbind(data.frame(V = names(coef(m_base)), E = as.numeric(coef(m_base)), M = "Baseline"),
                 data.frame(V = names(coef(m_svc)), E = as.numeric(coef(m_svc)), M = "SVCC"))
p6 <- ggplot(df_comb, aes(x = reorder(V, E), y = E, fill = M)) + geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() + labs(title = "Simulated Comparison of Estimated Coefficients") + theme_minimal()
ggsave(file.path(output_dir, "plot6_coefs.png"), p6)

# P7: Residual Spatial Mapping
sim_df$res_dev <- residuals(m_svc, type = "deviance")
p7 <- ggplot(sim_df, aes(x = x, y = y, color = res_dev)) + geom_point() + scale_color_gradient2() + 
  labs(title = "Residual Spatial Mapping") + theme_minimal()
ggsave(file.path(output_dir, "plot7_res_map.png"), p7)

# P8: Risk Density
p8 <- ggplot(sim_df, aes(x = wind_speed, y = time)) + stat_density_2d(aes(fill = ..level..), geom = "polygon") + 
  facet_wrap(~cut(x, 2)) + labs(title = "Bivariate Risk Density") + theme_minimal()
ggsave(file.path(output_dir, "plot8_density.png"), p8)

# P9: Selection Comparison
p9 <- ggplot(melt(table6, id = "Model"), aes(x = variable, y = value, fill = Model)) + geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "Selection Metric Comparison") + theme_minimal()
ggsave(file.path(output_dir, "plot9_selection.png"), p9)

# P10: Interaction Field
grid <- expand.grid(wd = seq(0, 2*pi, length = 50), x = seq(0, 1, length = 10))
grid$z <- sin(grid$wd) * grid$x
p10 <- ggplot(grid, aes(x = wd, y = x, fill = z)) + geom_tile() + coord_polar() + scale_fill_viridis_c(option = "B") + 
  labs(title = "Circular-Spatial Interaction Field") + theme_minimal()
ggsave(file.path(output_dir, "plot10_interaction.png"), p10)

# P11: VIF Resolution
vif_raw <- vif(lm(time ~ wind_speed_s + sin_wd_s + raw_x_sin + raw_y_cos, data = sim_df))
vif_ortho <- vif(lm(time ~ wind_speed_s + sin_wd_s + x_sin + y_cos, data = sim_df))
vif_viz <- data.frame(P = rep(names(vif_raw), 2), V = c(as.numeric(vif_raw), as.numeric(vif_ortho)), 
                      Mode = rep(c("Raw", "Ortho"), each = length(vif_raw)))
p11 <- ggplot(vif_viz, aes(x = P, y = V, fill = Mode)) + geom_bar(stat = "identity", position = "dodge") + 
  geom_hline(yintercept = 10, color = "red", linetype = "dashed") + labs(title = "VIF Resolution") + theme_minimal()
ggsave(file.path(output_dir, "plot11_vif.png"), p11)

cat("\nAnalysis Complete. Results saved in:", output_dir, "\n")

