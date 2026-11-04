# AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data

## Academic Metadata & Project Context
- **Student Register Number:** 24BCE0551
- **Track:** Multivariate and Time-Series Detection
- **Assigned Machine Learning Method:** Principal Component Analysis (PCA)-Based Multivariate Anomaly Detector
- **Primary Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*
- **Primary Evaluation Strategy:** Dual Statistical Process Control (Squared Prediction Error / SPE & Hotelling's $T^2$), Time-Series EWMA Monitoring, and Randomized Monte Carlo Synthetic Testing (200 trials per magnitude across $1.0\sigma, 2.0\sigma, 3.0\sigma, 4.0\sigma$).
- **Assigned Preprocessing Protocol:** Physical Plausibility Screening, SimpleImputer (median, train-only), StandardScaler (train-only), and Chronological Partitioning (70% Train, 15% Val, 15% Test).
- **Official Raw Dataset:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls` (Preserved Unmodified, SHA-256 Verified).

---

## Complete Project Directory Map
```text
Chiller-Anomaly-Detection-Official/
│
├── data/
│   ├── raw/
│   │   └── official_dataset/
│   │       └── Analyse Chillers_AI Course.xls     # Authoritative raw Excel workbook (preserved)
│   ├── processed/
│   │   ├── selected_features.json                 # 28 approved thermodynamic features config
│   │   ├── split_manifest.json                    # Chronological split boundaries & statistics
│   │   ├── train_features.csv                     # Standardized cleaned training matrix (358 x 28)
│   │   ├── validation_features.csv                # Standardized cleaned validation matrix (92 x 28)
│   │   ├── test_features.csv                      # Standardized cleaned testing matrix (67 x 28)
│   │   ├── train_features_raw.csv                 # Raw uncleaned features for historical traceability
│   │   ├── validation_features_raw.csv            # Raw uncleaned validation features
│   │   ├── test_features_raw.csv                  # Raw uncleaned test features
│   │   ├── train_metadata.csv                     # Training metadata (Timestamp, Chiller Tag, Loads)
│   │   ├── validation_metadata.csv                # Validation metadata
│   │   ├── test_metadata.csv                      # Testing metadata
│   │   ├── imputer.joblib                         # Fitted SimpleImputer (median, train-only)
│   │   ├── scaler.joblib                          # Fitted StandardScaler (train-only)
│   │   └── pca_model.joblib                       # Fitted PCA Detector (k = 12 components)
│   └── synthetic/
│       ├── spike_test_copy.csv                    # Scenario A test dataset (sudden spikes)
│       ├── drift_test_copy.csv                    # Scenario B test dataset (incipient thermal drift)
│       ├── relationship_breaking_test_copy.csv    # Scenario C test dataset (T-P decoupling)
│       └── monte_carlo_trials_dataset.csv         # 800-trial randomized benchmark dataset
│
├── notebooks/
│   └── Chiller_PCA_Anomaly_Detection.ipynb        # Fully executed master Jupyter Notebook (Phases 1-8)
│
├── src/
│   ├── data_inspection.py                         # Phase 1: Dataset profiling & workbook inspection
│   ├── eda_analysis.py                            # Phase 2: EDA, quality checks & correlation matrix
│   ├── preprocessing.py                           # Phase 3: Plausibility masking, imputation & scaling
│   ├── pca_detector.py                            # Phase 4: Clean PCA training, loadings & k sensitivity
│   ├── anomaly_detection_phase5.py                # Phase 5: Dual MSPC (SPE + T^2), EWMA & thresholding
│   ├── synthetic_experiments_phase6.py            # Phase 6: Monte Carlo benchmark & physical scenarios
│   ├── final_results_phase7.py                    # Phase 7: Synthesis, 12 final figures & evidence tables
│   ├── build_notebook.py                          # Automation script to assemble master notebook
│   └── assemble_submission.py                     # Final submission packaging and verification
│
├── results/
│   ├── figures/
│   │   ├── eda/                                   # 7 Phase 2 EDA figures (300 DPI)
│   │   ├── preprocessing/                         # 3 Phase 3 Preprocessing figures (300 DPI)
│   │   ├── pca/                                   # 6 Phase 4 PCA figures (300 DPI)
│   │   ├── phase5/                                # 7 Phase 5 Detection & EWMA figures (300 DPI)
│   │   ├── phase6/                                # 7 Phase 6 Monte Carlo & Scenario figures (300 DPI)
│   │   └── final/                                 # 12 Consolidated publication figures (300 DPI)
│   │
│   └── metrics/
│       ├── eda/                                   # 10 Phase 2 EDA metric tables
│       ├── preprocessing/                         # 6 Phase 3 Preprocessing & Cleaning tables
│       │   ├── masked_anomalies_log.csv           # Record of all 17 masked typos
│       │   ├── raw_vs_cleaned_comparison.csv      # Side-by-side variance correction table
│       │   ├── imputation_statistics.csv          # Median imputation tracking
│       │   ├── training_feature_variance.csv      # Positive variance verification
│       │   ├── scaling_parameters.csv             # Mean and scale parameters
│       │   └── preprocessing_validation.csv       # 10-point anti-leakage audit
│       ├── pca_explained_variance.csv             # Phase 4 full 28-component variance spectrum
│       ├── pca_component_loadings.csv             # Phase 4 28x12 eigenvector loadings matrix
│       ├── pca_k_sensitivity_analysis.csv         # Phase 4 k sensitivity table (k in {6,8,10,12,14,16,17})
│       ├── pca_training_scores.csv                # Phase 4 training projection scores (358 x 12)
│       ├── phase5/                                # Phase 5 MSPC, threshold & per-chiller tables
│       ├── phase6/                                # Phase 6 Monte Carlo, confusion matrices & PR tables
│       └── final/                                 # Phase 7 Consolidated comparison & evidence tables
│
├── documentation/
│   ├── ai_assistance_log.md                       # Transparent academic integrity assistance log
│   ├── dataset_profile_working.md                 # Phase 1 factual working profile
│   ├── initial_data_dictionary.csv                # Phase 1 60-variable classified dictionary
│   ├── phase1_file_inventory.csv                  # Phase 1 repository file inventory
│   ├── phase1_decision_log.md                     # Phase 1 approved engineering decisions
│   ├── phase2_eda_working.md                      # Phase 2 factual EDA working report
│   ├── phase3_preprocessing_working.md            # Phase 3 factual preprocessing working report
│   ├── phase4_pca_working.md                      # Phase 4 factual PCA model working report
│   ├── model_configuration.md                     # Comprehensive model specifications
│   ├── evaluation_and_interpretation.md           # Research question evidence synthesis
│   └── final_submission_audit.md                  # Comprehensive verification audit report
│
├── presentation/                                  # Slide deck presentation materials
├── Case_Study_Report_Template.docx                # Formal academic case study report document
├── requirements.txt                               # Verified Python package dependencies
└── README.md                                      # Master comprehensive engineering documentation
```

---

## Summary of the 8 Methodological & Technical Enhancements

In accordance with rigorous academic and industrial standards, 8 comprehensive improvements were incorporated:

1. **Physical Plausibility Rules & Data Cleaning:**
   - Identified 16 extreme typo cells in Training (e.g., `Liquid Temp Comp 1` = 631.1 °C, 329.0 °C; `Discharge Pressure` = 88.0 kg/cm²; `Evaporator Water In` = 103 °C). In raw data, this inflated `Liquid Temp Comp 1` standard deviation 35.2× (from 1.00 °C to 35.33 °C), blinding PCA to real faults.
   - Defined physical thermodynamic bounds based on R134a refrigeration physics. Violating cells are masked to `NaN` and imputed with `SimpleImputer(strategy='median')` fitted strictly on training data.
   - Preserved complete transparency via `results/metrics/preprocessing/raw_vs_cleaned_comparison.csv` and `results/figures/preprocessing/raw_vs_cleaned_comparison.png`.

2. **Validation Outlier Masking & Definite Threshold Calibration:**
   - Row 86 (Sl.No 73, Chiller-1) had `Discharge Saturation Temp Comp 2 = 322 °C` ($z = 132.6$), producing $\text{SPE} \approx 14,984$ and corrupting the 99th percentile threshold to 1378.
   - Masked via physical bounds and imputed to median. Clean validation thresholds are definitively established: $\tau_{\text{SPE}, 95} = 4.2703$, $\tau_{\text{SPE}, 99} = 7.8858$. All "Provisional" wording eliminated.

3. **Randomized Multi-Trial Monte Carlo Benchmark (200 Trials per Magnitude):**
   - Replaced fixed hand-picked tests with a randomized benchmark of **800 total injections** across magnitudes $1.0\sigma, 2.0\sigma, 3.0\sigma, 4.0\sigma$ using `np.random.seed(42)`.
   - Empirically mapped the PCA sensitivity curve with 95% Wilson binomial confidence intervals: 1.5% detection at $1.0\sigma$ (normal operating variance), rising to 6.0% at $2.0\sigma$, 80.0% at $3.0\sigma$, and 89.5%–93.5% at $4.0\sigma$.

4. **100% Dynamic Calculation (Zero Hard-Coded Metrics):**
   - Eliminated all static strings and hard-coded values (`drift_delay_hours`, `Clean_Baseline_FP_Contribution`, etc.). All metrics, delays, and false alarm counts are dynamically derived directly from data arrays.

5. **Method Gaps Filled — Hotelling's $T^2$ & Time-Series EWMA Monitoring:**
   - **Hotelling's $T^2$ Statistic:** Monitors Mahalanobis distance inside the principal subspace ($T^2 = \sum t_j^2 / \lambda_j$), catching operating shifts that move parallel to principal components where SPE is orthogonal.
   - **Time-Series EWMA Layer:** Implemented Exponentially Weighted Moving Average ($S_t = \lambda \text{SPE}_t + (1-\lambda) S_{t-1}$ with $\lambda = 0.20$), accumulating gradual incipient thermal drift and fulfilling the "Time-Series Detection" track mandate.

6. **Component Selection ($k$) Sensitivity Analysis:**
   - With clean data, $k=6$ satisfies the strict Kaiser criterion ($\lambda \ge 1.0$, 87.18% variance).
   - Evaluated sensitivity table across $k \in \{6, 8, 10, 12, 14, 16, 17\}$.
   - Primary model $k=12$ achieves **97.46% variance** (vs 85.46% on noisy data), captures dual-compressor cross-coupling, and reduces validation mean SPE to 0.9108 while leaving 16 residual dimensions for orthogonal fault isolation.

7. **Reclassification of Baseline Test Excursions & Metric Upgrade:**
   - Reclassified the 2 clean test flags (Index 38 on Chiller-4, Index 54 on Chiller-5) as genuine candidate operational excursions: Chiller-4 had 10.9 °C condenser approach (design maximum is 3–5 °C), and Chiller-5 suffered EXV differential pressure collapse.
   - Demoted misleading accuracy; prioritized Precision, Recall, F1-Score, and generated PR / ROC curves.

8. **Fleet Mix Shift Documentation & Submission Hygiene:**
   - Documented the Chiller-1 composition shift (12.3% of train vs 50.7% of test) and pooled multi-unit assumptions.
   - Master Jupyter notebook executed end-to-end with all 16 code cells showing verified saved outputs.
   - All figures rendered with distinct line styles (`-`, `--`, `-.`, `:`), markers (`o`, `s`, `^`, `X`, `v`), and cross-hatching for black-and-white print legibility.

---

## Complete 28 Core Thermodynamic Features

| # | Feature Name in Dataset | Subsystem | Domain Meaning | Unit |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `Evaporator Water In Temperature Deg C` | Water Loop | Entering chilled water return temperature from building | °C |
| 2 | `Evaporator Water Out Temperature Deg C` | Water Loop | Leaving chilled water supply setpoint temperature | °C |
| 3 | `Condenser Water In Temperature Deg C` | Water Loop | Entering cooling tower water into condenser | °C |
| 4 | `Condenser Water Out Temperature Deg C` | Water Loop | Leaving condenser water returning to cooling tower | °C |
| 5 | `Discharge Temperature Deg C Comp 1` | Refrigerant | High-side refrigerant vapor discharge temp, Comp 1 | °C |
| 6 | `Discharge Temperature Deg C Comp 2` | Refrigerant | High-side refrigerant vapor discharge temp, Comp 2 | °C |
| 7 | `Suction Temperature Deg C` | Refrigerant | Common low-side refrigerant vapor suction temp | °C |
| 8 | `Chiler Lift Temperature Deg C Comp 1` | Refrigerant | Lift temperature differential ($T_{\text{dis}} - T_{\text{suc}}$), Comp 1 | °C |
| 9 | `Chiler Lift Temperature Deg C Comp 2` | Refrigerant | Lift temperature differential ($T_{\text{dis}} - T_{\text{suc}}$), Comp 2 | °C |
| 10 | `Suction Saturation Temperature Deg C` | Refrigerant | Saturation boiling point at suction pressure | °C |
| 11 | `Suction Superheat deg C` | Refrigerant | Suction gas superheat ($T_{\text{suc}} - T_{\text{suc\_sat}}$) | °C |
| 12 | `Evaporator Approach` | Heat Exchanger | Approach temperature difference at evaporator | °C (Inferred) |
| 13 | `Discharge Saturation Temperature Deg C Comp 1` | Refrigerant | Condensing saturation temp Comp 1 | °C |
| 14 | `Discharge Saturation Temperature Deg C Comp 2` | Refrigerant | Condensing saturation temp Comp 2 | °C |
| 15 | `Discharge Superheat Deg C Comp 1` | Refrigerant | Discharge vapor superheat Comp 1 | °C |
| 16 | `Discharge Superheat Deg C Comp 2` | Refrigerant | Discharge vapor superheat Comp 2 | °C |
| 17 | `Condenser Approach Comp 1` | Heat Exchanger | Approach temperature difference at condenser Comp 1 | °C (Inferred) |
| 18 | `Condenser Approach Comp 2` | Heat Exchanger | Approach temperature difference at condenser Comp 2 | °C (Inferred) |
| 19 | `Liquid Temperature Deg C Comp 1` | Refrigerant | Subcooled liquid refrigerant temperature circuit 1 | °C |
| 20 | `Liquid Temperature Deg C Comp 2` | Refrigerant | Subcooled liquid refrigerant temperature circuit 2 | °C |
| 21 | `Oil Temperature Deg C` | Lubrication | Compressor lubrication oil temperature | °C |
| 22 | `Discharge Pressure Kg /cm 2 Comp 1` | Refrigerant | High-side condensing refrigerant pressure Comp 1 | $\text{kg/cm}^2$ |
| 23 | `Discharge Pressure Kg /cm 2 Comp 2` | Refrigerant | High-side condensing refrigerant pressure Comp 2 | $\text{kg/cm}^2$ |
| 24 | `Suction Pressure Kg /cm 2` | Refrigerant | Common low-side evaporating refrigerant pressure | $\text{kg/cm}^2$ |
| 25 | `Chiller Lift Pressure Kg /cm 2 Comp 1` | Refrigerant | Pressure lift differential ($P_{\text{dis}} - P_{\text{suc}}$) Comp 1 | $\text{kg/cm}^2$ |
| 26 | `Chiller Lift Pressure Kg /cm 2 Comp 2` | Refrigerant | Pressure lift differential ($P_{\text{dis}} - P_{\text{suc}}$) Comp 2 | $\text{kg/cm}^2$ |
| 27 | `EXV Differential Pressure Kg /cm 2 Comp 1` | Refrigerant | Pressure drop across Electronic Expansion Valve 1 | $\text{kg/cm}^2$ |
| 28 | `EXV Differential Pressure Kg /cm 2 Comp 2` | Refrigerant | Pressure drop across Electronic Expansion Valve 2 | $\text{kg/cm}^2$ |

---

## Core Research Question Empirical Answer

> **"Can anomalies be detected from correlated temperature and pressure variables?"**
>
> **YES, EMPIRICALLY CONFIRMED.** Industrial chiller operations exhibit tight physical coupling between temperature and pressure loops ($r > +0.95$). PCA captures 97.46% of this covariance in a 12-dimensional principal subspace. When a physical fault breaks or violates these correlations, the observation cannot be represented by the principal components and is projected orthogonally into the residual subspace, elevating Squared Prediction Error (SPE) by 6× to 10× above the normal baseline. When combined with Hotelling's $T^2$ and an EWMA time-series filter, both abrupt sensor failures and incipient thermal drifts are reliably intercepted.

---

## Step-by-Step Execution Guide

To reproduce the entire experimental pipeline from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Phase 1: Data Inspection
python src/data_inspection.py

# 3. Phase 2: Exploratory Data Analysis & Correlation Matrix
python src/eda_analysis.py

# 4. Phase 3: Physical Plausibility Screening, Anti-Leakage Split & Scaling
python src/preprocessing.py

# 5. Phase 4: Clean PCA Model Development & k Sensitivity Analysis
python src/pca_detector.py

# 6. Phase 5: Dual MSPC (SPE + T^2), EWMA Layer & Validation Thresholds
python src/anomaly_detection_phase5.py

# 7. Phase 6: Monte Carlo Benchmark & Synthetic Scenarios
python src/synthetic_experiments_phase6.py

# 8. Phase 7: Consolidated Synthesis & Publication Figures
python src/final_results_phase7.py

# 9. Master Notebook Assembly & In-Place Execution
python src/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/Chiller_PCA_Anomaly_Detection.ipynb

# 10. Final Submission Archival
python src/assemble_submission.py
```

---

## Technical Viva Voce Defense Preparation

1. **How does PCA detect anomalies in correlated thermodynamic variables?**
   * PCA models normal operating correlation structure. When normal physics holds (e.g. higher discharge temperature corresponds to higher condensing pressure), observations project closely onto the $k$-dimensional principal subspace. When a fault occurs (such as sensor decalibration, refrigerant leak, or EXV failure), this correlation breaks, forcing the point into the residual subspace and producing an elevated Squared Prediction Error ($\text{SPE} = \|\mathbf{z} - \hat{\mathbf{z}}\|^2 > \tau$).

2. **Why is Hotelling's $T^2$ needed in addition to SPE?**
   * SPE only measures deviations *orthogonal* to the principal subspace (broken correlations). If an operating excursion moves *along* the correlated direction (e.g. an extreme operating load shift or severe plant-wide thermal overload), SPE remains small, but Hotelling's $T^2 = \sum t_j^2 / \lambda_j$ measures Mahalanobis distance *inside* the principal plane, triggering an alarm.

3. **Why is a time-series layer (EWMA) necessary?**
   * Point-by-point static SPE evaluates each hour independently. For subtle incipient drift ($< 2.5\sigma$), individual observations remain within normal variance. EWMA ($S_t = \lambda \text{SPE}_t + (1-\lambda) S_{t-1}$) accumulates consistent temporal drift, providing timely warning hours before critical threshold breach.

4. **How was data leakage prevented during model development?**
   * Splitting was performed chronologically on unique plant timestamps (first 70% Train, next 15% Validation, final 15% Test). Transformers (`SimpleImputer` and `StandardScaler`) and the PCA model were fitted **strictly on the Training partition**. The anomaly thresholds ($\tau_{\text{SPE}}, \tau_{T^2}, \tau_{\text{EWMA}}$) were calibrated strictly on the clean Validation partition and applied frozen to Test data with zero parameter adjustment.
