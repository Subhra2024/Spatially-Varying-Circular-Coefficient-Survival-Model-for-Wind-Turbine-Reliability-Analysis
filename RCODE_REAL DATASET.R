
# --- 1. Required Libraries ---
if (!require("survival")) install.packages("survival")
if (!require("survminer")) install.packages("survminer")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("pec")) install.packages("pec")
if (!require("car")) install.packages("car")

library(survival)
library(survminer)
library(ggplot2)
library(pec) 
library(car)

# --- 2. Setup ---
output_dir <- "Real_Data_Full_Results"
if (!dir.exists(output_dir)) dir.create(output_dir)
set.seed(42)

# =============================================================================
# 3. DATA LOADING & BASIS EXPANSION
# =============================================================================
cat("Step 1: Loading and Processing Dataset...\n")

# NOTE: For GitHub reproducibility, keep the CSV in the same directory as the script
real_df <- read.csv("C:/Users/SUBHRAJIT SAHA/Downloads/real_turbine_failure_data.csv")

# Basis Expansion for interaction terms (Spatial coordinates x Circular components)
real_df$x_sin_wd <- real_df$x * real_df$sin_wd
real_df$y_cos_wd <- real_df$y * real_df$cos_wd

# Fit Cox Models
# x=TRUE and y=TRUE are required for the IBS calculation in the 'pec' package
m_base <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd, 
                data = real_df, x=TRUE, y=TRUE)

m_svc  <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd + x_sin_wd + y_cos_wd, 
                data = real_df, x=TRUE, y=TRUE)

# =============================================================================
# 4. GENERATE 8 TABLES (Including Robustness)
# =============================================================================
cat("Step 2: Generating Tables...\n")

# --- TABLE 1: Predictive Performance (MAE & IBS) ---
ibs_res <- pec(list(Baseline=m_base, Proposed_SVC=m_svc), 
               formula=Surv(time, event)~1, data=real_df, reference=FALSE)

table1 <- data.frame(
  Metric = c("Mean Absolute Error (MAE)", "Integrated Brier Score (IBS)"),
  Baseline_Cox_Model = c(mean(abs(real_df$time - predict(m_base, type="expected"))),
                         ibs_res$AppErr$Baseline[length(ibs_res$AppErr$Baseline)]),
  Proposed_SVC_Model = c(mean(abs(real_df$time - predict(m_svc, type="expected"))),
                         ibs_res$AppErr$Proposed_SVC[length(ibs_res$AppErr$Proposed_SVC)])
)
write.csv(table1, file.path(output_dir, "table1_mae_ibs.csv"), row.names = FALSE)

# --- TABLE 2: Advanced Multicollinearity Diagnostics ---
vif_mod <- lm(time ~ wind_speed + sin_wd + cos_wd + x_sin_wd + y_cos_wd, data=real_df)
X_scaled <- scale(model.matrix(vif_mod), center=FALSE, scale=sqrt(colSums(model.matrix(vif_mod)^2)))
cond_indices <- max(svd(X_scaled)$d) / svd(X_scaled)$d
corr_intercept <- summary(vif_mod, correlation=TRUE)$correlation[1, 2:6] 

table2 <- data.frame(
  Predictor = names(vif(vif_mod)),
  VIF = as.numeric(vif(vif_mod)),
  Condition_Index = cond_indices[2:6],
  Corr_with_Intercept = as.numeric(corr_intercept)
)
write.csv(table2, file.path(output_dir, "table2_collinearity_diagnostics.csv"), row.names = FALSE)

# --- TABLE 3: Descriptive Statistics ---
write.csv(summary(real_df), file.path(output_dir, "table3_descriptive.csv"))

# --- TABLE 4: Baseline Cox Summary ---
write.csv(as.data.frame(summary(m_base)$coefficients), file.path(output_dir, "table4_baseline_coefs.csv"))

# --- TABLE 5: Proposed SVC Cox Summary ---
write.csv(as.data.frame(summary(m_svc)$coefficients), file.path(output_dir, "table5_svc_coefs.csv"))

# --- TABLE 6: Model Selection Metrics (AIC, BIC, LogLik) ---
table6 <- data.frame(
  Model = c("Baseline Cox", "Proposed SVC"),
  AIC = c(AIC(m_base), AIC(m_svc)),
  BIC = c(BIC(m_base), BIC(m_svc)),
  LogLikelihood = c(as.numeric(logLik(m_base)), as.numeric(logLik(m_svc)))
)
write.csv(table6, file.path(output_dir, "table6_selection_criteria.csv"), row.names = FALSE)

# =============================================================================
# 5. GENERATE 6 PLOTS
# =============================================================================
cat("Step 3: Generating Plots...\n")

# 5.1 Spatial Distribution
p1 <- ggplot(real_df, aes(x=x, y=y, fill=time)) + 
  geom_point(size=4, shape=21, color="black") + 
  scale_fill_viridis_c(name="Months to Failure") + 
  theme_minimal() + labs(title="Spatial Distribution of Turbine Failures")
ggsave(file.path(output_dir, "plot1_spatial.png"), p1, width=8, height=6)

# 5.2 Wind Speed
p2 <- ggplot(real_df, aes(x=wind_speed)) + 
  geom_histogram(fill="steelblue", color="white", bins=15) + 
  theme_minimal() + labs(title="Operational Wind Speed Profile", x="Wind Speed (m/s)")
ggsave(file.path(output_dir, "plot2_wind.png"), p2, width=8, height=6)

# 5.3 Rose Plot
real_df$wd_deg <- (atan2(real_df$sin_wd, real_df$cos_wd) * 180 / pi) %% 360
p3 <- ggplot(real_df, aes(x=wd_deg)) + 
  geom_histogram(fill="orange", color="black", bins=16) + 
  coord_polar() + theme_minimal() + labs(title="Circular Wind Direction Profile")
ggsave(file.path(output_dir, "plot3_wind_rose.png"), p3, width=8, height=8)

# 5.4 Kaplan-Meier
png(file.path(output_dir, "plot4_km_curve.png"), width=800, height=600)
plot(survfit(Surv(time, event)~1, data=real_df), col="red", lwd=2, 
     main="Kaplan-Meier Survival Curve", xlab="Time (Months)", ylab="Probability")
dev.off()

# 5.5 Cumulative Hazard
p5 <- ggplot(basehaz(m_svc), aes(x=time, y=hazard)) + 
  geom_step(color="darkgreen", size=1) + 
  theme_minimal() + labs(title="Estimated Cumulative Baseline Hazard")
ggsave(file.path(output_dir, "plot5_hazard.png"), p5, width=8, height=6)

# 5.6 Coefficient Comparison
coef_long <- rbind(
  data.frame(Term = names(coef(m_base)), Estimate = coef(m_base), Model = "Baseline"),
  data.frame(Term = names(coef(m_svc)), Estimate = coef(m_svc), Model = "SVC Proposed")
)
p6 <- ggplot(coef_long, aes(x=Term, y=Estimate, fill=Model)) +
  geom_bar(stat="identity", position=position_dodge(), color="black") +
  coord_flip() + theme_minimal() + labs(title="Estimated Model Coefficients")
ggsave(file.path(output_dir, "plot6_coefficients.png"), p6, width=10, height=6)

# =============================================================================
# 6. EMPIRICAL ROBUSTNESS & SENSITIVITY ANALYSIS
# =============================================================================
cat("Step 4: Running Empirical Robustness Checks...\n")

# --- 6.1 Stability across Spatial Resolutions (M) ---
m_alt <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd + 
                 I(x^2 * sin_wd) + I(y^2 * cos_wd), data = real_df)

robust_table <- data.frame(
  Test_Scenario = c("Standard SVCC (Linear Interaction)", "High-Complexity SVCC (Quadratic Interaction)"),
  AIC = c(AIC(m_svc), AIC(m_alt)),
  LogLik = c(as.numeric(logLik(m_svc)), as.numeric(logLik(m_alt)))
)
write.csv(robust_table, file.path(output_dir, "table7_empirical_robustness.csv"), row.names = FALSE)

# --- 6.2 Data Partitioning Stability (Bootstrapping Performance) ---
cat("Running Bootstrapped Stability Check (100 iterations)...\n")
boot_ibs <- c()
for(i in 1:100) {
  sample_idx <- sample(1:nrow(real_df), replace = TRUE)
  boot_df <- real_df[sample_idx, ]
  m_boot <- coxph(Surv(time, event) ~ wind_speed + sin_wd + cos_wd + x_sin_wd + y_cos_wd, 
                  data = boot_df, x=TRUE, y=TRUE)
  ibs_boot <- pec(list(SVC=m_boot), formula=Surv(time, event)~1, data=boot_df, reference=FALSE, verbose=FALSE)
  boot_ibs[i] <- ibs_boot$AppErr$SVC[length(ibs_boot$AppErr$SVC)]
}

boot_results <- data.frame(
  Metric = "IBS Stability (Mean ± SD)",
  Value = paste0(round(mean(boot_ibs), 4), " ± ", round(sd(boot_ibs), 4))
)
write.csv(boot_results, file.path(output_dir, "table8_bootstrap_stability.csv"), row.names = FALSE)

cat("\nAnalysis Complete. All 8 Tables and 6 Plots have been saved to:", output_dir, "\n")
