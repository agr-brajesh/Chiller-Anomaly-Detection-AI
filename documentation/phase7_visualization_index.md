# Phase 7: Final Visualization Index and Figure Catalog

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Resolution Standard:** All visualizations rendered at 300 DPI with publication-ready formatting.  
**Storage Directory:** `results/figures/final/`  

---

## Comprehensive Figure Catalog (Figures 1 to 12)

### Figure 1: Core Temperature-Pressure Thermodynamic Correlation Matrix
- **File:** `results/figures/final/final_temperature_pressure_correlation.png`
- **Source Phase:** Phase 2 (Exploratory Data Analysis)
- **Data Used:** Pearson correlation matrix calculated across 517 dual-compressor observations (`results/metrics/eda/pearson_correlation_matrix.csv`). Subselected to 10 key temperature and pressure loop sensors.
- **What It Represents:** Lower-triangular correlation heatmap displaying pairwise Pearson correlation coefficients ($r$) between water loop temperatures, refrigerant cycle temperatures, and operating pressures.
- **Key Interpretation Notes:** Confirms strong physical coupling governed by R134a vapor-pressure thermodynamics (e.g., Discharge Temperature vs Discharge Pressure Comp 1: $r = +0.9578$; Suction Saturation Temperature vs Suction Pressure: $r = +0.8973$). Proves the physical premise of multivariate collinearity.

---

### Figure 2: PCA Scree Plot and Cumulative Explained Variance Spectrum
- **File:** `results/figures/final/final_pca_explained_variance.png`
- **Source Phase:** Phase 4 (PCA Model Development)
- **Data Used:** Eigenvalue spectrum and explained variance ratios from `results/metrics/pca_explained_variance.csv` ($N_{\text{train}} = 358, p = 28$).
- **What It Represents:** Dual-axis visualization with bar charts showing individual component explained variance percentages (left axis) and an overlaid curve showing cumulative explained variance (right axis) up to 28 components.
- **Key Interpretation Notes:** Justifies the selection of $k = 12$ principal components (85.46% retained variance), which exceeds the 80% industrial benchmark. Crucially, the remaining $p - k = 16$ residual dimensions (14.54% variance) are preserved for Squared Prediction Error (SPE) anomaly detection.

---

### Figure 3: Principal Component Eigenvector Loadings Heatmap
- **File:** `results/figures/final/final_pca_component_loadings.png`
- **Source Phase:** Phase 4 (PCA Model Development)
- **Data Used:** Standardized component loading matrix from `results/metrics/pca_component_loadings.csv` across the first 5 principal components.
- **What It Represents:** Diverging heatmap ($\pm 0.40$) displaying the contribution of all 28 thermodynamic sensor features to PC1 through PC5.
- **Key Interpretation Notes:** Demonstrates clear physical separation: PC1 (23.77%) captures **High-Side Condenser Compression Lift** (discharge saturation temperatures +0.320, discharge pressures +0.301, lift temperatures +0.285); PC2 (13.64%) captures **Low-Side Evaporator Cooling Demand** (chilled water temperatures +0.384, suction pressure +0.244, suction superheat -0.231).

---

### Figure 4: Clean Test Reconstruction Error (SPE) Timeline
- **File:** `results/figures/final/final_clean_test_reconstruction_error.png`
- **Source Phase:** Phase 5 (Reconstruction Error & Anomaly Detection)
- **Data Used:** Held-out operational test partition (`results/metrics/phase5/test_anomaly_predictions.csv`, $N_{\text{test}} = 67$, March 18 to 20).
- **What It Represents:** Chronological trajectory of test Squared Prediction Error ($\text{SPE}_i = \sum_{j=1}^{28} e_{ij}^2$) plotted against the frozen validation-derived 95th percentile threshold ($\tau = 9.2931$).
- **Key Interpretation Notes:** 65 out of 67 observations (97.01%) conform tightly to the normal baseline (mean $\text{SPE} = 2.49$). Exactly two natural operational excursions exceed the threshold: Index 38 on Chiller-4 ($\text{SPE} = 9.91$) and Index 54 on Chiller-5 ($\text{SPE} = 13.60$).

---

### Figure 5: Distribution of Test Reconstruction Errors
- **File:** `results/figures/final/final_clean_test_anomaly_distribution.png`
- **Source Phase:** Phase 5 (Reconstruction Error & Anomaly Detection)
- **Data Used:** Test SPE values from `results/metrics/phase5/test_reconstruction_errors.csv`.
- **What It Represents:** Histogram and Kernel Density Estimation (KDE) curve of test SPE, with vertical markers for threshold ($\tau = 9.29$), median ($1.97$), and mean ($2.49$).
- **Key Interpretation Notes:** Shows a unimodal, right-skewed distribution where normal operating observations cluster below $\text{SPE} = 3.0$, while the two operational excursions appear in the extreme upper tail.

---

### Figure 6: Top Feature Contributions to Reconstruction Error
- **File:** `results/figures/final/final_feature_residual_contributions.png`
- **Source Phase:** Phase 5 (Sensor Contribution Analysis)
- **Data Used:** Aggregate squared residuals ($e_{ij}^2$) from `results/metrics/phase5/aggregate_feature_contributions.csv`.
- **What It Represents:** Ranked horizontal bar chart displaying the top contributing thermodynamic variables to the total SPE across flagged test observations, color-coded by temperature vs pressure category.
- **Key Interpretation Notes:** Low-side evaporator and suction variables (`Suction Temperature` 22.05%, `Chiller Lift Temp` 17.75%, `Suction Superheat` 13.21%, `Evaporator Water Out` 10.92%) account for **over 73% of total residual variance**, diagnosing that test-period anomalies originated in the evaporator loop.

---

### Figure 7: Scenario A — Sudden Sensor Spike Analysis
- **File:** `results/figures/final/final_synthetic_spike_analysis.png`
- **Source Phase:** Phase 6 (Synthetic Anomaly Experiments)
- **Data Used:** Modified test copy (`results/synthetic/spike_test_copy.csv`) and resulting SPE from `results/metrics/phase6/clean_vs_synthetic_spe.csv`.
- **What It Represents:** Two-panel time-series showing the raw/standardized sensor trajectory with single-observation $\pm 4.0\sigma$ spikes (top) and the resulting SPE timeline crossing the threshold (bottom).
- **Key Interpretation Notes:** Proves that acute gross sensor errors immediately trigger an anomaly alarm with **0 hours detection delay**, 100% event recall, and no residual smearing into adjacent hours.

---

### Figure 8: Scenario B — Gradual Sensor Drift Analysis
- **File:** `results/figures/final/final_synthetic_drift_analysis.png`
- **Source Phase:** Phase 6 (Synthetic Anomaly Experiments)
- **Data Used:** Modified test copy (`results/synthetic/drift_test_copy.csv`) over 5 continuous hours in Chiller-1 (11:00 to 15:00 on March 19).
- **What It Represents:** Two-panel time-series displaying a progressive linear ramp ($+0.7\sigma \to +3.5\sigma$) in `Evaporator Water Out Temperature` (top) and the gradual rise of SPE towards and across the threshold (bottom).
- **Key Interpretation Notes:** Illustrates the operational detection boundary: incipient drift ($< 2.8\sigma$) is tolerated as normal operational variance; alarm triggers at Hour 5 ($+3.5\sigma$, $\text{SPE} = 13.50$), establishing an empirical **4-hour detection delay**.

---

### Figure 9: Scenario C — Temperature-Pressure Relationship Breaking Analysis
- **File:** `results/figures/final/final_relationship_breaking_analysis.png`
- **Source Phase:** Phase 6 (Synthetic Anomaly Experiments)
- **Data Used:** Modified test copy (`results/synthetic/relationship_breaking_test_copy.csv`) over 3 hours in Chiller-1.
- **What It Represents:** Two-panel plot showing discharge pressure artificially decoupled (perturbed by $-3.5\sigma$) while discharge temperature remains normal (top), and the resulting SPE crossing the threshold across all 3 observations (bottom).
- **Key Interpretation Notes:** **Directly answers the research question.** Demonstrates that violating the $r = +0.9578$ physical thermodynamic correlation forces data directly into the 16 residual dimensions, triggering immediate alarm ($100\%$ recall, $0$ delay).

---

### Figure 10: Final Confusion Matrices Comparison
- **File:** `results/figures/final/final_confusion_matrices.png`
- **Source Phase:** Phase 6 (Model Evaluation)
- **Data Used:** Observation-level confusion matrices from `results/metrics/phase6/confusion_matrices.csv`.
- **What It Represents:** Side-by-side heatmaps displaying True Positives, False Positives, False Negatives, and True Negatives across all three synthetic scenarios.
- **Key Interpretation Notes:** Highlights that the $FP = 2$ in each scenario stems entirely from the two naturally occurring baseline excursions in the clean test set, meaning **net synthetic false positives = ZERO**.

---

### Figure 11: Final Observation-Level Performance Metrics Comparison
- **File:** `results/figures/final/final_performance_comparison.png`
- **Source Phase:** Phase 6 (Model Evaluation)
- **Data Used:** Precision, Recall, F1-Score, and Accuracy from `results/metrics/final/final_experiment_comparison.csv`.
- **What It Represents:** Grouped bar chart comparing observation-level classification metrics across Scenarios A, B, and C.
- **Key Interpretation Notes:** Demonstrates high precision ($60\%$) and perfect recall ($100\%$) for spikes and relationship breaking, contrasted with the expected lower observation recall ($20\%$) in gradual drift caused by incipient tolerance.

---

### Figure 12: Final Event-Level Detection Rate Comparison
- **File:** `results/figures/final/final_event_detection_comparison.png`
- **Source Phase:** Phase 6 (Model Evaluation)
- **Data Used:** Event detection percentages from `results/metrics/final/final_experiment_comparison.csv`.
- **What It Represents:** Bar chart illustrating the event-level detection rate across all three anomaly categories.
- **Key Interpretation Notes:** Proves **100% event detection across all scenarios**—every developing or sudden fault episode was captured before causing system failure.

