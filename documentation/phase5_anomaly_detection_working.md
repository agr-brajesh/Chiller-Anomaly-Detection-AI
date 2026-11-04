# Phase 5: Reconstruction Error Derivation and PCA-Based Anomaly Detection

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Research Question:** Can anomalies be detected from correlated temperature and pressure variables?  
**Phase Status:** Completed (Activities A through G)  
**Date:** 2026-11-04  

---

## 1. Phase 5 Implementation Overview

In Phase 5, anomaly detection is formally realized using the reconstruction capability of the trained Principal Component Analysis model ($k = 12$ components, $p = 28$ thermodynamic features).

### Core Mathematical Concepts
1. **Projection and Reconstruction in Standardized Feature Space:**
   $$\mathbf{z}_i = \mathbf{x}_i \mathbf{V}_k$$
   $$\hat{\mathbf{x}}_i = \mathbf{z}_i \mathbf{V}_k^T$$
   where $\mathbf{x}_i \in \mathbb{R}^{28}$ is the standardized observation vector, $\mathbf{V}_k \in \mathbb{R}^{28 \times 12}$ is the orthogonal projection matrix of the 12 retained eigenvectors, and $\hat{\mathbf{x}}_i \in \mathbb{R}^{28}$ is the reconstructed observation vector.
2. **Squared Prediction Error (SPE / Q-Statistic):**
   $$\mathbf{e}_i = \mathbf{x}_i - \hat{\mathbf{x}}_i$$
   $$\text{SPE}_i = \|\mathbf{e}_i\|_2^2 = \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2 = \sum_{j=1}^{p} e_{ij}^2$$
   *Note:* In accordance with industrial multivariate statistical process control (MSPC), SPE is calculated as the **sum of squared residuals**, not the mean squared error. (Mean squared error is simply $\text{SPE} / 28$).
3. **Threshold Calibration on Validation Partition Only:**
   The anomaly detection threshold $\tau$ is determined strictly from the held-out validation set ($N_{\text{val}} = 92$), enforcing zero test-set leakage.
4. **Test Set Flagging:**
   For each test observation $\mathbf{x}_{\text{test}, i}$:
   $$y_i = \begin{cases} 1 & \text{if } \text{SPE}_i > \tau \quad (\text{Potential Anomaly}) \\ 0 & \text{if } \text{SPE}_i \le \tau \quad (\text{Normal Operation}) \end{cases}$$

---

## 2. Reused Artifacts and Verification

All models, transformers, and split datasets were loaded from previously verified disk artifacts:
- **PCA Model:** `data/processed/pca_model.joblib` ($k = 12$ retained components, $p = 28$ input features, fitted on $N_{\text{train}} = 358$).
- **Imputer:** `data/processed/imputer.joblib` (`SimpleImputer(strategy='median')`, fitted on Train only).
- **Scaler:** `data/processed/scaler.joblib` (`StandardScaler()`, fitted on Train only).
- **Validation Features:** `data/processed/validation_features.csv` ($92 \times 28$).
- **Test Features:** `data/processed/test_features.csv` ($67 \times 28$).
- **Feature Names:** 28 thermodynamic features matching `selected_features.json` in exact order.

*Verification Check:* No models were retrained, no transformers refitted, and no hyperparameter tuning was performed on test data.

---

## 3. Reconstruction Error Statistics

Reconstruction errors were calculated across all three partitions to provide baseline comparison:

| Partition | Sample Size ($N$) | Mean SPE | Median SPE | Std Dev | Min SPE | Max SPE | 95th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (Reference)** | 358 | 4.0709 | 2.1466 | 9.8732 | 0.4431 | 129.4120 | 11.2341 |
| **Validation** | 92 | 165.8660 | 2.2505 | 1561.4320 | 0.6124 | 14983.8655 | 9.2931 |
| **Test (Untouched)** | 67 | 2.4913 | 1.9749 | 2.1492 | 0.6425 | 13.5997 | 5.4975 |

### Key Observations on Distribution Dynamics
1. **Median Consistency:** The median SPE values across all three partitions are remarkably consistent:
   - Train Median: $2.1466$
   - Validation Median: $2.2505$
   - Test Median: $1.9749$
   This confirms that the vast majority of observations across all partitions conform to the same underlying thermodynamic subspace captured by the 12 principal components.
2. **The Validation Tail Outlier:** Validation mean ($165.87$) and maximum ($14983.87$) are heavily skewed by a single extreme recording error:
   - Observation: Chiller - 1 at `2026-03-18 18:00:00` (Raw Sl.No 73).
   - Sensor: `Discharge Saturation Temperature Deg C Comp 2` was entered as $322.0\text{ }^\circ\text{C}$ instead of $32.2\text{ }^\circ\text{C}$ (missing decimal point typo).
   - Standardized z-score for this single feature: $z = +132.62$.
   - Individual squared residual for this feature alone: $e_{ij}^2 \approx 14,892.4$.
   - Total SPE: $14,983.87$.
   This empirically validates that the PCA detector responds with extreme sensitivity to thermodynamic violations.

---

## 4. Validation Anomaly Threshold Analysis

Threshold candidates were evaluated exclusively on the validation set ($N_{\text{val}} = 92$):

| Candidate Rule | Threshold SPE ($\tau$) | Val Obs Above | Val % Above | Effective Tail Points | Order-Statistic Rank | Risk / Trade-Off Analysis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **90.0th Percentile** | 5.0538 | 10 | 10.87% | 10 | Top 10 of 92 | High sensitivity; flags ~10% of normal steady-state operation as false alarms. |
| **95.0th Percentile (Recommended)** | **9.2931** | **5** | **5.43%** | **5** | **Top 5 of 92** | **Balanced industrial baseline; anchored by ~5 observations, providing statistical stability.** |
| **97.5th Percentile** | 11.9014 | 3 | 3.26% | 3 | Top 3 of 92 | More conservative; determined by only 2–3 observations, subject to small-sample variance. |
| **99.0th Percentile** | 1378.3543 | 1 | 1.09% | 1 | Top 1 of 92 | **Invalidated by outlier;** pulled up to 1378.35 by the single operator logging typo ($322^\circ\text{C}$). |

### Statistical Rationale for the Recommended Primary Threshold
1. **Order-Statistic Sensitivity in Small Samples:** With $N_{\text{val}} = 92$, a 99th percentile represents a tail probability where only $92 \times (1 - 0.99) = 0.92$ observations exist. Thus, the 99th percentile is determined entirely by the maximum observation. Because the validation set contains an extreme human logging typo ($322^\circ\text{C}$), the 99th percentile is distorted by orders of magnitude ($\tau = 1378.35$), which would completely blind the detector to genuine operational anomalies.
2. **Robustness of the 95th Percentile:** In contrast, the 95th percentile ($\tau = 9.2931$) is determined by the 87th and 88th sorted observations, effectively isolating the normal operational envelope while resisting extreme single-point contamination.
3. **Threshold Recommendation:** **$\tau = 9.2931$** is recommended as the provisional primary threshold.

---

## 5. Test Partition Anomaly Evaluation

Applying the frozen validation threshold ($\tau = 9.2931$) to the held-out test partition ($N_{\text{test}} = 67$):

| Metric | Computed Value | Description |
| :--- | :---: | :--- |
| **Total Test Observations** | 67 | March 18, 20:00 to March 20, 08:00 |
| **Flagged Potential Anomalies** | 2 | Observations where $\text{SPE} > 9.2931$ |
| **Unflagged Normal Observations** | 65 | Observations where $\text{SPE} \le 9.2931$ |
| **Flagged Percentage** | **2.99%** | Approximately 3% of operational test time |
| **Mean Test SPE** | 2.4913 | Well below threshold |
| **Median Test SPE** | 1.9749 | Comparable to training median (2.15) |
| **Maximum Test SPE** | 13.5997 | Observed at Chiller-5 |
| **Minimum Test SPE** | 0.6425 | Observed at Chiller-3 |

### Detailed Breakdown of Flagged Test Observations

#### Anomaly 1: Chiller - 5 at 2026-03-20 02:00:00
- **Test Index:** 54
- **Reconstruction Error:** $\text{SPE} = 13.5997$ (Highest error in test set)
- **Top Contributing Variables:**
  1. `Chiler Lift Temperature Deg C Comp 1`: $27.79\%$ of SPE ($z = -4.56$)
  2. `Suction Saturation Temperature Deg C`: $13.94\%$ of SPE ($z = -0.18$)
  3. `Suction Superheat deg C`: $13.80\%$ of SPE ($z = +0.43$)
  4. `Evaporator Water Out Temperature Deg C`: $12.23\%$ of SPE ($z = +2.17$)
  5. `Discharge Temperature Deg C Comp 2`: $7.41\%$ of SPE ($z = -0.70$)
- **Physical Interpretation:** At 2:00 AM, Compressor 1 on Chiller-5 experienced a severe loss of temperature lift ($z = -4.56$) while the chilled water leaving temperature rose to an unusually warm state ($z = +2.17$). This pattern represents a thermodynamic mismatch: the chiller was unable to produce normal temperature lift despite elevated chilled water exit temperatures, consistent with severe compressor unloading or internal hot gas bypass.

#### Anomaly 2: Chiller - 4 at 2026-03-19 17:00:00
- **Test Index:** 38
- **Reconstruction Error:** $\text{SPE} = 9.9126$ (Exceeds threshold 9.2931)
- **Top Contributing Variables:**
  1. `Suction Temperature Deg C`: $46.90\%$ of SPE ($z = +2.65$)
  2. `Suction Superheat deg C`: $12.39\%$ of SPE ($z = -1.20$)
  3. `Oil Temperature Deg C`: $9.90\%$ of SPE ($z = -1.15$)
  4. `Evaporator Water Out Temperature Deg C`: $9.13\%$ of SPE ($z = -1.00$)
  5. `Chiler Lift Temperature Deg C Comp 1`: $3.99\%$ of SPE ($z = +0.43$)
- **Physical Interpretation:** At 17:00 on March 19, Suction Temperature ran abnormally high ($z = +2.65$) accounting for nearly half of the total squared residual ($46.9\%$). Simultaneously, suction superheat and oil temperatures were suppressed, indicating anomalous refrigerant vapor heating prior to compressor intake.

---

## 6. Sensor Contribution Analysis

Across all flagged test observations, sensor-wise squared residuals ($e_{ij}^2$) were aggregated to evaluate which physical measurement groups account for reconstruction failures:

| Sensor Variable | Domain Category | Total Squared Residual | Relative Contribution (%) |
| :--- | :---: | :---: | :---: |
| **Suction Temperature Deg C** | Temperature | 4.8876 | **22.05%** |
| **Chiler Lift Temperature Deg C Comp 1** | Temperature | 3.9358 | **17.75%** |
| **Suction Superheat deg C** | Temperature | 2.9275 | **13.21%** |
| **Evaporator Water Out Temperature Deg C** | Temperature | 2.4208 | **10.92%** |
| **Suction Saturation Temperature Deg C** | Temperature | 2.0406 | **9.20%** |
| **Discharge Temperature Deg C Comp 2** | Temperature | 1.3444 | 6.06% |
| **Oil Temperature Deg C** | Temperature | 1.1448 | 5.16% |
| **Suction Pressure Kg /cm 2** | Refrigerant Pressure | 0.8124 | **3.66%** |
| **Discharge Pressure Kg /cm 2 Comp 2** | Refrigerant Pressure | 0.6945 | **3.13%** |
| **EXV Differential Pressure Kg /cm 2 Comp 1**| Refrigerant Pressure | 0.5120 | **2.31%** |

### Thermodynamic Insight
Low-side evaporator and suction line variables (`Suction Temperature`, `Chiller Lift Temp`, `Suction Superheat`, `Evaporator Water Out`) account for **over 73% of the total reconstruction error** in flagged observations. High-side condenser pressure and discharge temperatures remained largely stable, indicating that abnormal operational transients in the test period originated in the refrigeration heat absorption (evaporation/suction) loop rather than heat rejection (condenser).

---

## 7. Chiller-Specific Diagnostic Breakdown

Evaluating the distribution of test reconstruction errors across individual chiller units:

| Chiller ID | Test Observations | Flagged Count | Flagged % | Mean SPE | Median SPE | Max SPE | Min SPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chiller - 5** | 12 | 1 | 8.33% | 2.5968 | 1.3678 | 13.5997 | 0.6734 |
| **Chiller - 4** | 15 | 1 | 6.67% | 3.1006 | 2.2441 | 9.9126 | 1.1060 |
| **Chiller - 1** | 34 | 0 | 0.00% | 2.2697 | 2.0143 | 5.6991 | 0.8616 |
| **Chiller - 3** | 6 | 0 | 0.00% | 2.0123 | 2.1205 | 2.9728 | 0.6425 |
| **Chiller - 2** | 0 | 0 | N/A | N/A | N/A | N/A | N/A |

### Diagnostic Findings
1. **Chiller - 1 Stability:** Despite having the highest number of observations in the test set ($N = 34$, comprising over 50% of test records), Chiller - 1 had **0 flagged observations** and a maximum SPE of only $5.70$. This demonstrates that Chiller - 1 operates within a highly stable thermodynamic regime that is well-represented by the pooled PCA model.
2. **Disproportionate Flags in Chillers 4 & 5:** The two flagged anomalies occurred in Chiller - 4 and Chiller - 5. Both units exhibit higher mean and maximum SPE values than Chillers 1 and 3.
3. **Temporal Isolation:** The two flagged events occurred at distinct, non-consecutive timestamps (March 19 at 17:00 and March 20 at 02:00, separated by 9 hours). They represent isolated operational excursions rather than persistent, multi-hour sensor drift.

---

## 8. Limitations and Scope Boundaries

1. **Absence of Ground-Truth Fault Labels:** No verified ground-truth maintenance logs or fault records are available in the raw operational dataset. Therefore, flagged observations must be interpreted strictly as **statistical anomalies or unusual operational states**, not confirmed mechanical failures.
2. **Supervised Metrics Exclusion:** In compliance with scientific integrity standards, supervised metrics (Precision, Recall, ROC-AUC, F1-Score) cannot be calculated on real operational test data. Supervised metrics will be evaluated strictly in Phase 6 using controlled synthetic anomaly injections.
3. **Small Sample Uncertainty:** With $N_{\text{val}} = 92$, empirical percentile thresholds carry inherent sampling variance. While the 95th percentile is reasonably stable, small shifts in plant operating envelopes over seasons could alter threshold calibration.

---

## 9. Generated Artifacts Inventory

### Metric CSVs & JSON Configuration (`results/metrics/phase5/`)
- `validation_reconstruction_errors.csv` ($92\text{ rows}$)
- `test_reconstruction_errors.csv` ($67\text{ rows}$)
- `validation_threshold_comparison.csv` (Candidate percentiles 90%, 95%, 97.5%, 99%)
- `threshold_configuration.json` (Provisional primary threshold specification)
- `test_anomaly_predictions.csv` ($67\text{ rows}$ with anomaly flags)
- `test_anomaly_summary.csv` (Overall test summary statistics)
- `top_potential_anomalies.csv` (Top 10 highest-SPE test observations)
- `feature_residual_contributions.csv` ($67 \times 28$ feature contributions)
- `aggregate_feature_contributions.csv` (Ranked feature contributions across flagged points)
- `chiller_anomaly_summary.csv` (Fleet-wise breakdown across Chillers 1–5)

### Figures (`results/figures/phase5/`, 300 DPI)
- `validation_threshold_analysis.png`: Validation SPE histogram with candidate threshold lines.
- `test_reconstruction_timeline.png`: Test SPE timeline plotted against the frozen threshold.
- `test_reconstruction_distribution.png`: Histogram and KDE of test reconstruction errors.
- `test_potential_anomalies.png`: Discrete scatter plot of normal vs flagged points over time.
- `feature_residual_contributions.png`: Horizontal bar chart of top 10 contributing sensors.
- `chiller_reconstruction_comparison.png`: Box and jitter plots of SPE across individual chillers.

