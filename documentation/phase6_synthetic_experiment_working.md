# Phase 6: Synthetic Anomaly Experiments and Model Evaluation

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*  
**Phase Status:** Completed (Activities A through H)  
**Date:** 2026-11-04  

---

## 1. Experimental Objectives & Design Rationale

Because real-world operational datasets rarely contain verified ground-truth fault labels, supervised classification metrics (Precision, Recall, F1-Score, Detection Delay) cannot be evaluated on unlabelled operational data. In Phase 6, we design controlled, mathematically transparent synthetic perturbation experiments to evaluate the detection sensitivity, promptness, and robustness of the frozen PCA detector ($\tau = 9.2931$).

### Key Design Principles:
1. **Unchanged Clean Reference:** The original clean test partition ($N_{\text{test}} = 67, p = 28$) is preserved intact. All synthetic scenarios are generated as separate, isolated copies.
2. **Standardized Perturbation Space:** Modifications are injected directly into copies of the standardized test feature matrix. An injection of $\Delta z = \pm 4.0$ corresponds exactly to a change of $4.0$ training standard deviations ($\pm 4.0\sigma$) in physical units.
3. **No Retraining / Refitting:** The PCA model ($k = 12$) and anomaly threshold ($\tau = 9.2931$) remain frozen from Phases 4 and 5.
4. **Targeting Clean Baselines:** Synthetic modifications were placed at candidate timestamps and chillers where the original clean-test SPE was well below the threshold ($\text{SPE} \ll 9.2931$), enabling precise measurement of incremental detection.
5. **Within-Chiller Continuity:** Multi-observation events were strictly confined within the continuous chronological run of a single chiller.

---

## 2. The Three Synthetic Scenarios & Anomaly Injection Manifest

| Event ID | Scenario Category | Chiller ID | Timestamp Range | Duration | Target Sensor Variable | Perturbation Magnitude | Direction | Modified Observations | Clean Baseline SPE |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **SPIKE_01** | Scenario A: Sudden Spike | Chiller - 1 | 2026-03-19 09:00:00 | 1 hr | `Discharge Temperature Deg C Comp 1` | $+4.0\sigma$ | Positive | 1 | 0.8654 |
| **SPIKE_02** | Scenario A: Sudden Spike | Chiller - 4 | 2026-03-19 13:00:00 | 1 hr | `Discharge Pressure Kg /cm 2 Comp 1` | $+4.0\sigma$ | Positive | 1 | 1.7660 |
| **SPIKE_03** | Scenario A: Sudden Spike | Chiller - 1 | 2026-03-19 22:00:00 | 1 hr | `Suction Temperature Deg C` | $-4.0\sigma$ | Negative | 1 | 1.2880 |
| **DRIFT_01** | Scenario B: Gradual Drift | Chiller - 1 | 2026-03-19 11:00 to 15:00 | 5 hrs | `Evaporator Water Out Temperature Deg C` | $+0.7\sigma$ to $+3.5\sigma$ (Linear) | Positive Ramp | 5 | 1.42 to 2.60 |
| **REL_BREAK_01** | Scenario C: Relationship Breaking | Chiller - 1 | 2026-03-20 06:00 to 08:00 | 3 hrs | `Discharge Pressure Kg /cm 2 Comp 1` | $-3.5\sigma$ | Negative Perturbation | 3 | 1.52 to 1.70 |

---

## 3. Observation-Level Evaluation Results

Ground-truth binary labels ($y_{\text{true}} = 1$ for artificially modified points, $0$ for unmodified points) were compared against detector predictions ($\hat{y} = 1$ if $\text{SPE} > \tau = 9.2931$):

| Scenario | Total Obs ($N$) | Positives (Injected) | Negatives (Unmodified) | TP | FP | FN | TN | Precision | Recall | F1-Score | Accuracy | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario A: Sudden Spikes** | 67 | 3 | 64 | 3 | 2 | 0 | 62 | **0.6000** | **1.0000** | **0.7500** | **0.9701** | 0.0312 |
| **Scenario B: Gradual Drift** | 67 | 5 | 62 | 1 | 2 | 4 | 60 | **0.3333** | **0.2000** | **0.2500** | **0.9104** | 0.0323 |
| **Scenario C: Rel Breaking** | 67 | 3 | 64 | 3 | 2 | 0 | 62 | **0.6000** | **1.0000** | **0.7500** | **0.9701** | 0.0312 |

### Essential Nuance: Why False Positives Equal Exactly 2 Across All Scenarios
In the clean test baseline (Phase 5), exactly 2 observations naturally exceeded the threshold $\tau = 9.2931$:
1. Index 38: Chiller - 4 at `2026-03-19 17:00:00` ($\text{SPE} = 9.9126$)
2. Index 54: Chiller - 5 at `2026-03-20 02:00:00` ($\text{SPE} = 13.5997$)
Because synthetic ground-truth assigns label $0$ to all observations that were not artificially modified, these two naturally occurring baseline operational excursions are classified as "False Positives" ($FP = 2$).
**Net Synthetic False Positives across all three scenarios: ZERO.** No unmodified normal operating observation was falsely pushed across the threshold by synthetic perturbations.

---

## 4. Event-Level Detection & Temporal Delay

While observation-level metrics evaluate every single row, industrial plant operators evaluate anomaly detection at the **event level** (whether an ongoing fault condition is caught before causing equipment shutdown or damage):

| Scenario Category | Total Injected Events | Detected Events | Missed Events | Event Detection Rate (%) | Incremental Event Detection Rate (%) | Detection Delay (Hours) | Earliest Detection Timestamp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario A: Sudden Spikes** | 3 | 3 | 0 | **100.0%** | **100.0%** | **0 hours** (Immediate) | 2026-03-19 09:00:00 |
| **Scenario B: Gradual Drift** | 1 | 1 | 0 | **100.0%** | **100.0%** | **4 hours** (Breached at $+3.5\sigma$) | 2026-03-19 15:00:00 |
| **Scenario C: Rel Breaking** | 1 | 1 | 0 | **100.0%** | **100.0%** | **0 hours** (Immediate on 1st obs) | 2026-03-20 06:00:00 |

### Observation-Level vs Event-Level Dynamics in Gradual Drift
- In Scenario B, the drift offset increased linearly: $+0.7\sigma \to +1.4\sigma \to +2.1\sigma \to +2.8\sigma \to +3.5\sigma$.
  - At $+0.7\sigma$ (Hour 1): $\text{SPE} = 3.0557$ (Within normal bounds)
  - At $+1.4\sigma$ (Hour 2): $\text{SPE} = 4.4772$ (Elevated, but below threshold)
  - At $+2.1\sigma$ (Hour 3): $\text{SPE} = 7.1634$ (Approaching threshold)
  - At $+2.8\sigma$ (Hour 4): $\text{SPE} = 9.0071$ (Near critical boundary $9.2931$)
  - At $+3.5\sigma$ (Hour 5): $\text{SPE} = 13.4988$ (**Threshold breached!**)
- **Insight:** Observation-level recall is $1/5 = 20.0\%$ because early incipient drift is statistically indistinguishable from allowable plant operational tolerance. However, **event-level recall is 100.0%** because the developing drift was caught as soon as it exceeded normal physical boundaries, providing operators with actionable advance warning.

---

## 5. Direct Answer to the Assigned Research Question

**Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*

### Empirical Proof from Scenario C (Relationship Breaking):
In Phase 2 EDA, `Discharge Temperature Deg C Comp 1` and `Discharge Pressure Kg /cm 2 Comp 1` exhibited an extremely strong physical correlation ($r = +0.9578, \rho = +0.7641$), reflecting refrigerant R134a vapor-pressure thermodynamics.
In Scenario C, we decoupled this relationship:
- `Discharge Temperature Deg C Comp 1` was left untouched at normal operating levels (~$48.5^\circ\text{C}$).
- `Discharge Pressure Kg /cm 2 Comp 1` was perturbed by $-3.5\sigma$ (simulating a sensor calibration loss or refrigerant leak).
- **Result:**
  - Clean baseline SPE: $1.5208, 1.6968, 1.7012$.
  - Injected SPE: $\mathbf{9.5416, 9.5761, 9.7430}$ (All 3 observations breached the threshold $\tau = 9.2931$).
  - Detection rate: **100.0% (3/3 observations, 1/1 event)**.
  - Detection delay: **0 hours (immediate detection on the first affected observation)**.

**Scientific Conclusion:** When two physically correlated thermodynamic variables are artificially decoupled, the observed data point violates the low-dimensional covariance structure captured by the $k = 12$ principal components. The projection into the 16 residual dimensions explodes quadratically, driving the Squared Prediction Error (SPE) across the anomaly threshold. **This provides conclusive, verifiable empirical proof answering the research question affirmatively.**

---

## 6. Generated Artifacts Inventory

### Synthetic Datasets & Ground Truth (`results/synthetic/`)
- `spike_test_copy.csv`: Modified feature matrix for Scenario A ($67 \times 28$).
- `drift_test_copy.csv`: Modified feature matrix for Scenario B ($67 \times 28$).
- `relationship_breaking_test_copy.csv`: Modified feature matrix for Scenario C ($67 \times 28$).
- `spike_ground_truth.csv`: Binary ground-truth labels for Scenario A ($67\text{ rows}$).
- `drift_ground_truth.csv`: Binary ground-truth labels for Scenario B ($67\text{ rows}$).
- `relationship_breaking_ground_truth.csv`: Binary ground-truth labels for Scenario C ($67\text{ rows}$).

### Evaluation Metrics (`results/metrics/phase6/`)
- `anomaly_injection_manifest.csv`: Detailed manifest of all 5 synthetic events.
- `spike_metrics.csv`: Observation-level performance metrics for Scenario A.
- `drift_metrics.csv`: Observation-level performance metrics for Scenario B.
- `relationship_breaking_metrics.csv`: Observation-level performance metrics for Scenario C.
- `overall_metrics_comparison.csv`: Comparative summary across all 3 scenarios.
- `confusion_matrices.csv`: Breakdown of TP, FP, FN, TN and baseline contributions.
- `event_detection_results.csv`: Event-level detection rates and detection delays.
- `clean_vs_synthetic_spe.csv`: Observation-by-observation SPE comparison ($67 \times 9$).

### Publication-Quality Figures (`results/figures/phase6/`, 300 DPI)
- `synthetic_spike_analysis.png`: Sensor reading and resulting SPE timeline for sudden spikes.
- `synthetic_drift_analysis.png`: Sensor trajectory, linear ramp, and threshold breach for gradual drift.
- `relationship_breaking_analysis.png`: Decoupling of discharge temp and press with resulting SPE breach.
- `spe_comparison.png`: Simultaneous SPE trajectories of clean baseline vs all 3 synthetic scenarios.
- `confusion_matrix_comparison.png`: Heatmaps of confusion matrices for all 3 scenarios.
- `performance_comparison.png`: Bar chart comparing Precision, Recall, F1-Score, and Accuracy.
- `event_detection_comparison.png`: Bar chart comparing event detection rates (100% across all scenarios).

