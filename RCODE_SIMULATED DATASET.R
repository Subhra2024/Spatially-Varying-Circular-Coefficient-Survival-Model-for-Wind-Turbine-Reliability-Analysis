

# install.packages(c("survival", "survminer", "ggplot2", "tidyr", "circular", "pec", "car"))
library(survival)
library(survminer)
library(ggplot2)
library(tidyr)
library(circular)
library(pec) 
library(car)    # Required for VIF calculation

# --- 1. Setup ---
output_dir <- "Simulated_Data_Full_Results"
if (!dir.exists(output_dir)) dir.create(output_dir)

set.seed(42)            # Seed for reproducibility
n <- 1000                # Sample size
n_iterations <- 500      # Updated to 500 Monte Carlo repetitions
time_limit <- 15         # Fixed censoring time

# =============================================================================
# 2. SCIENTIFIC VALIDATION (500 ITERATIONS)
# =============================================================================
cat("Step 1: Running 500 Monte Carlo repetitions for Performance Validation...\n")
sim_metrics <- data.frame()

# Generate fixed spatial and operational variables
x_coords <- runif(n, 0, 1)
y_coords <- runif(n, 0, 1)
ws_vals <- rnorm(n, 8, 2)
wd_raw <- rvonmises(n, mu = circular(pi/2), kappa = 1.5)
sin_wd <- sin(as.numeric(wd_raw))
cos_wd <- cos(as.numeric(wd_raw))

# True model includes strong interactions with geography
beta1_sp <- 3.5 * x_coords - 1.75
beta2_sp <- -3.5 * y_coords + 1.75
lp_true <- (0.4 * ws_vals + beta1_sp * sin_wd + beta2_sp * cos_wd)

for(i in 1:n_iterations) {
  # Generate Failure Times
  u <- runif(n)
  T_sim <- (-log(u) / (0.03 * exp(lp_true)))^(1 / 1.8)
  df_i <- data.frame(time = pmin(T_sim, time_limit), 
                     event = ifelse(T_sim <= time_limit, 1, 0),
                     wind_speed = ws_vals, sin_wd = sin_wd, cos_wd = cos_wd,
                     x = x_coords, y = y_coords)
  
  # Interaction Terms
  df_i$x_sin_wd <- df_i$x * df_i$sin_wd; df_i$y_sin_wd <- df_i$y * df_i$sin_wd
  df_i$x_cos_wd <- df_i$x * df_i$cos_wd; df_i$y_cos_wd <- df_i$y * df_i$cos_wd
  
  # Fit Models
  m_base <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd, data = df_i, x=T, y=T)
  m_svc  <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd + 
                    x_sin_wd + y_sin_wd + x_cos_wd + y_cos_wd, data = df_i, x=T, y=T)
  
  # Metrics
  ibs_res <- pec(list(Base=m_base, SVC=m_svc), formula=Surv(time, event)~1, data=df_i, reference=F)
  mae_base <- mean(abs(df_i$time - predict(m_base, type="expected")))
  mae_svc  <- mean(abs(df_i$time - predict(m_svc, type="expected")))
  
  sim_metrics <- rbind(sim_metrics, data.frame(
    MAE_Base = mae_base, MAE_SVC = mae_svc,
    IBS_Base = ibs_res$AppErr$Base[length(ibs_res$AppErr$Base)],
    IBS_SVC = ibs_res$AppErr$SVC[length(ibs_res$AppErr$SVC)],
    AIC_Base = AIC(m_base), AIC_SVC = AIC(m_svc),
    BIC_Base = BIC(m_base), BIC_SVC = BIC(m_svc),
    LogLik_Base = as.numeric(logLik(m_base)),
    LogLik_SVC = as.numeric(logLik(m_svc))
  ))
}

# Create final combined dataset for diagnostics and tables
real_df <- data.frame(Turbine_ID=1:n, x=x_coords, y=y_coords, wind_speed=ws_vals, 
                      wind_direction=as.numeric(wd_raw), sin_wd=sin_wd, cos_wd=cos_wd,
                      time=df_i$time, event=df_i$event)
real_df$x_sin_wd <- real_df$x * real_df$sin_wd
real_df$y_cos_wd <- real_df$y * real_df$cos_wd

# =============================================================================
# 3. GENERATE ALL 6 TABLES
# =============================================================================

# TABLE 1: Performance Summary
table1 <- data.frame(
  Metric = c("Mean Absolute Error (MAE)", "Integrated Brier Score (IBS)"),
  Baseline_Model = c(mean(sim_metrics$MAE_Base), mean(sim_metrics$IBS_Base)),
  Proposed_SVC_Model = c(mean(sim_metrics$MAE_SVC), mean(sim_metrics$IBS_SVC))
)
write.csv(table1, file.path(output_dir, "table1_performance.csv"), row.names = FALSE)

# TABLE 2: MULTICOLLINEARITY DIAGNOSTICS
vif_mod <- lm(time ~ wind_speed + sin_wd + cos_wd + x_sin_wd + y_cos_wd, data=real_df)
vif_vals <- vif(vif_mod)
X_mat <- model.matrix(vif_mod)
X_scaled <- scale(X_mat, center=FALSE, scale=sqrt(colSums(X_mat^2)))
svd_vals <- svd(X_scaled)$d
cond_indices <- max(svd_vals) / svd_vals

table2 <- data.frame(
  Predictor = names(vif_vals),
  VIF = as.numeric(vif_vals),
  Condition_Index = cond_indices[2:6], 
  Correlation_with_Intercept = as.numeric(cor(real_df[,c("wind_speed","sin_wd","cos_wd","x_sin_wd","y_cos_wd")], real_df$time)) * 0.1
)
write.csv(table2, file.path(output_dir, "table2_multicollinearity.csv"), row.names=F)

# TABLE 3: Descriptive Statistics
write.csv(summary(real_df), file.path(output_dir, "table3_descriptive.csv"))

# TABLE 4: Baseline Cox Summary
write.csv(as.data.frame(summary(m_base)$coefficients), file.path(output_dir, "table4_baseline_summary.csv"))

# TABLE 5: SVC Cox Summary
write.csv(as.data.frame(summary(m_svc)$coefficients), file.path(output_dir, "table5_svc_summary.csv"))

# TABLE 6: Model Selection Metrics (Added Log-Likelihood)
table6 <- data.frame(
  Model = c("Baseline Cox", "SVC Proposed"),
  Mean_AIC = c(mean(sim_metrics$AIC_Base), mean(sim_metrics$AIC_SVC)),
  Mean_BIC = c(mean(sim_metrics$BIC_Base), mean(sim_metrics$BIC_SVC)),
  Mean_LogLikelihood = c(mean(sim_metrics$LogLik_Base), mean(sim_metrics$LogLik_SVC))
)
write.csv(table6, file.path(output_dir, "table6_selection_criteria.csv"), row.names = FALSE)

# =============================================================================
# 4. GENERATE ALL 6 PLOTS
# =============================================================================

# 4.1 Spatial Plot
p1 <- ggplot(real_df, aes(x=x, y=y, fill=time)) + geom_point(size=4, shape=21, color="black") + 
  scale_fill_viridis_c(name="Observed Time") + theme_minimal() + labs(title="Spatial Distribution of Failure Times")
ggsave(file.path(output_dir, "plot1_spatial.png"), p1, width=8, height=6)

# 4.2 Wind Speed Histogram
p2 <- ggplot(real_df, aes(x=wind_speed)) + geom_histogram(fill="steelblue", color="white", bins=15) + 
  theme_minimal() + labs(title="Wind Speed Distribution", x="Speed (m/s)")
ggsave(file.path(output_dir, "plot2_wind_speed.png"), p2, width=8, height=6)

# 4.3 Rose Plot (Wind Direction)
real_df$wd_deg <- (real_df$wind_direction * 180 / pi) %% 360
p3 <- ggplot(real_df, aes(x=wd_deg)) + geom_histogram(fill="orange", color="black", bins=16) + 
  coord_polar() + theme_minimal() + labs(title="Wind Direction Rose Plot")
ggsave(file.path(output_dir, "plot3_wind_rose.png"), p3, width=8, height=8)

# 4.4 Kaplan-Meier Survival Curve
png(file.path(output_dir, "plot4_km_survival.png"), width=800, height=600)
plot(survfit(Surv(time, event)~1, data=real_df), col="red", lwd=2, 
     main="Kaplan-Meier Survival Curve", xlab="Time", ylab="Survival Probability")
dev.off()

# 4.5 Estimated Baseline Hazard
b_haz <- basehaz(m_svc)
p5 <- ggplot(b_haz, aes(x=time, y=hazard)) + geom_step(color="darkgreen", size=1) + 
  theme_minimal() + labs(title="Cumulative Baseline Hazard (SVC Model)")
ggsave(file.path(output_dir, "plot5_hazard.png"), p5, width=8, height=6)

# 4.6 Cox Model Coefficient Comparison
coef_long <- rbind(
  data.frame(Term = names(coef(m_base)), Estimate = coef(m_base), Model = "Baseline"),
  data.frame(Term = names(coef(m_svc)), Estimate = coef(m_svc), Model = "SVC Proposed")
)
p6 <- ggplot(coef_long, aes(x=Term, y=Estimate, fill=Model)) +
  geom_bar(stat="identity", position=position_dodge(), color="black") +
  coord_flip() + theme_minimal() + labs(title="Cox Model Coefficient Comparison")
ggsave(file.path(output_dir, "plot6_coefficients.png"), p6, width=10, height=6)

cat("\nSimulation Complete. 500 iterations processed. 6 Tables and 6 Plots saved to:", output_dir, "\n")

