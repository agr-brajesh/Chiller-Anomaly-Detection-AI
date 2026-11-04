# Phase 3 Working Document: Data Preprocessing and Anti-Data-Leakage Protocol

**Document Purpose:** Factual engineering notes recording the execution of Phase 3 preprocessing, chronological partitioning, imputation, scaling, and anti-data-leakage validation.

---

## 1. Selected Feature List (28 Core Thermodynamic Features)

The primary PCA input space is restricted to the **28 core thermodynamic variables** approved in Phase 1 and verified in Phase 2:

### Temperature Variables (21 Features — all in °C)
1. `Evaporator Water In Temperature Deg C`: Entering chilled water return temperature.
2. `Evaporator Water Out Temperature Deg C`: Leaving chilled water supply setpoint temperature.
3. `Condenser Water In Temperature Deg C`: Entering condenser water from cooling tower.
4. `Condenser Water Out Temperature Deg C`: Leaving condenser water returning to cooling tower.
5. `Discharge Temperature Deg C Comp 1`: High-pressure refrigerant vapor discharge temp, Comp 1.
6. `Discharge Temperature Deg C Comp 2`: High-pressure refrigerant vapor discharge temp, Comp 2.
7. `Suction Temperature Deg C`: Low-pressure refrigerant vapor suction temp entering compressors.
8. `Chiler Lift Temperature Deg C Comp 1`: Temperature lift differential ($T_{\text{discharge}} - T_{\text{suction}}$), Comp 1.
9. `Chiler Lift Temperature Deg C Comp 2`: Temperature lift differential ($T_{\text{discharge}} - T_{\text{suction}}$), Comp 2.
10. `Suction Saturation Temperature Deg C`: Saturation boiling temp corresponding to suction pressure.
11. `Suction Superheat deg C`: Superheat of suction gas ($T_{\text{suction}} - T_{\text{suction\_sat}}$).
12. `Evaporator Approach`: Approach temperature at evaporator ($\text{Deg C (Inferred)}$).
13. `Discharge Saturation Temperature Deg C Comp 1`: Condensing saturation temp Comp 1.
14. `Discharge Saturation Temperature Deg C Comp 2`: Condensing saturation temp Comp 2.
15. `Discharge Superheat Deg C Comp 1`: Discharge gas superheat Comp 1.
16. `Discharge Superheat Deg C Comp 2`: Discharge gas superheat Comp 2.
17. `Condenser Approach Comp 1`: Approach temperature at condenser Comp 1 ($\text{Deg C (Inferred)}$).
18. `Condenser Approach Comp 2`: Approach temperature at condenser Comp 2 ($\text{Deg C (Inferred)}$).
19. `Liquid Temperature Deg C Comp 1`: Subcooled liquid refrigerant temperature circuit 1.
20. `Liquid Temperature Deg C Comp 2`: Subcooled liquid refrigerant temperature circuit 2.
21. `Oil Temperature Deg C`: Compressor lubrication oil temperature.

### Pressure Variables (7 Features — all in kg/cm²)
22. `Discharge Pressure Kg /cm 2 Comp 1`: High-side condensing refrigerant pressure Comp 1.
23. `Discharge Pressure Kg /cm 2 Comp 2`: High-side condensing refrigerant pressure Comp 2.
24. `Suction Pressure Kg /cm 2`: Low-side evaporating refrigerant pressure.
25. `Chiller Lift Pressure Kg /cm 2 Comp 1`: Pressure lift differential ($P_{\text{discharge}} - P_{\text{suction}}$) Comp 1.
26. `Chiller Lift Pressure Kg /cm 2 Comp 2`: Pressure lift differential ($P_{\text{discharge}} - P_{\text{suction}}$) Comp 2.
27. `EXV Differential Pressure Kg /cm 2 Comp 1`: Pressure drop across Electronic Expansion Valve 1.
28. `EXV Differential Pressure Kg /cm 2 Comp 2`: Pressure drop across Electronic Expansion Valve 2.

### Excluded Operational Variables (Preserved in Separate Metadata)
- `Sl.No`, `CHILLER TAG`, `Timestamp`, `Date`, `Time`
- `Running Load % Comp 1`, `Running Load % Comp 2`
- `Motor Current Amps Comp 1`, `Motor Current Amps Comp 2`
- *Artifact Saved:* [`data/processed/selected_features.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/selected_features.json)

---

## 2. Observation Structure & Chronological Splitting Methodology

### Multi-Chiller Concurrency & Anti-Leakage Rule
- The dataset records 5 physical chillers operating concurrently across 222 unique hourly timestamps from `2026-03-11 00:00:00` to `2026-03-20 08:00:00`.
- To prevent temporal data leakage, partitioning was determined strictly on **unique chronological timestamps**:
  - All observations logged at the same plant hour remain in the same partition.
  - Zero observations from the same timestamp are split across training, validation, or testing.

### Partition Sizes & Time Boundaries
1. **Training Partition (First 70% of unique hours = 155 hours):**
   - Observation Count: **358 observations (69.25%)**
   - Time Span: `2026-03-11 00:00:00` to `2026-03-17 10:00:00`
   - Fleet Representation: Chiller-1: 67, Chiller-2: 82, Chiller-3: 79, Chiller-4: 76, Chiller-5: 54
2. **Validation Partition (Next 15% of unique hours = 33 hours):**
   - Observation Count: **92 observations (17.80%)**
   - Time Span: `2026-03-17 11:00:00` to `2026-03-18 19:00:00`
   - Fleet Representation: Chiller-1: 17, Chiller-2: 0 (switched off), Chiller-3: 25, Chiller-4: 33, Chiller-5: 17
3. **Testing Partition (Final 15% of unique hours = 34 hours):**
   - Observation Count: **67 observations (12.96%)**
   - Time Span: `2026-03-18 20:00:00` to `2026-03-20 08:00:00`
   - Fleet Representation: Chiller-1: 12, Chiller-2: 0, Chiller-3: 16, Chiller-4: 22, Chiller-5: 17
- **Total Partitioned Rows:** $358 + 92 + 67 = \mathbf{517}$ rows (100% accounted for).
- *Artifact Saved:* [`data/processed/split_manifest.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/split_manifest.json)
- *Visualization Saved:* [`results/figures/preprocessing/train_validation_test_distribution.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/preprocessing/train_validation_test_distribution.png)

---

## 3. Missing-Value Imputation Procedure

- **Algorithm:** `SimpleImputer(strategy="median")`
- **Fitting Rule:** Fitted strictly on the training partition ($N_{\text{train}} = 358$).
- **Transformation:** Applied to $X_{\text{train}}$, $X_{\text{val}}$, and $X_{\text{test}}$ using the fitted training medians.
- **Empirical Imputation Counts:**
  - Training set missing cells imputed: **0**
  - Validation set missing cells imputed: **0**
  - Test set missing cells imputed: **0**
  - *Note:* The historical dataset is 100% complete. The imputer is retained and persisted in the pipeline for operational robustness during testing and real-time streaming.
- *Artifact Saved:* [`data/processed/imputer.joblib`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/imputer.joblib)
- *Metrics Saved:* [`results/metrics/preprocessing/imputation_statistics.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/preprocessing/imputation_statistics.csv)

---

## 4. Zero-Variance & Feature Quality Check in Training Set

- **Variance Calculation:** Computed across all 28 features using $X_{\text{train}}$ only ($N = 358$).
- **Constant Features Detected:** **0** (All 28 features possess non-zero sample variance in training).
- **Lowest Variance Features in Training:**
  - `Suction Pressure Kg /cm 2`: $\sigma_{\text{train}}^2 = 0.0102$, $\sigma_{\text{train}} = 0.1010\text{ kg/cm}^2$ (normal tight regulation).
  - `Condenser Approach Comp 1`: $\sigma_{\text{train}}^2 = 0.3872$, $\sigma_{\text{train}} = 0.6223\text{ }^\circ\text{C}$.
- *Conclusion:* No features are degenerate or require removal. All 28 features can be safely standardized without division by zero.
- *Metrics Saved:* [`results/metrics/preprocessing/training_feature_variance.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/preprocessing/training_feature_variance.csv)

---

## 5. Feature Standardization (StandardScaler)

- **Mathematical Formula:**
  $$Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Fitting Rule:** `StandardScaler` fitted strictly on imputed training observations ($N = 358$).
- **Transformation:** $X_{\text{train}}$, $X_{\text{val}}$, and $X_{\text{test}}$ transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
- **Verification of Transformed Training Features:**
  - Max absolute mean: $4.62 \times 10^{-15} \approx 0.0$ (passed).
  - Mean standard deviation: $1.0014 \approx 1.0$ (passed).
- *Artifact Saved:* [`data/processed/scaler.joblib`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/scaler.joblib)
- *Metrics Saved:* [`results/metrics/preprocessing/scaling_parameters.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/preprocessing/scaling_parameters.csv)
- *Visualization Saved:* [`results/figures/preprocessing/scaling_comparison.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/preprocessing/scaling_comparison.png)

---

## 6. Preprocessing Validation & Anti-Data-Leakage Audit

All 18 anti-data-leakage criteria passed the validation audit:

| Validation Criterion | Expected | Observed | Audit Status |
| :--- | :--- | :--- | :--- |
| **Feature Space Dimensionality** | 28 columns | 28 columns | **PASS** |
| **Train Partition Count** | 358 rows | 358 rows | **PASS** |
| **Validation Partition Count** | 92 rows | 92 rows | **PASS** |
| **Test Partition Count** | 67 rows | 67 rows | **PASS** |
| **Total Observations Accounted For** | 517 rows | 517 rows | **PASS** |
| **Missing Values (Train / Val / Test)** | 0 / 0 / 0 | 0 / 0 / 0 | **PASS** |
| **Infinite Values (Train / Val / Test)** | 0 / 0 / 0 | 0 / 0 / 0 | **PASS** |
| **Training Scaled Mean** | $< 10^{-4}$ | $4.62 \times 10^{-15}$ | **PASS** |
| **Training Scaled Std** | $1.0000$ | $1.0014$ | **PASS** |
| **Temporal Boundary: Train End < Val Start** | True | True (`2026-03-17 10:00` < `11:00`) | **PASS** |
| **Temporal Boundary: Val End < Test Start** | True | True (`2026-03-18 19:00` < `20:00`) | **PASS** |
| **Chiller Identity Excluded from PCA Inputs** | True | True (retained in metadata only) | **PASS** |
| **Timestamp Excluded from PCA Inputs** | True | True (retained in metadata only) | **PASS** |
| **Raw Excel Source Preservation** | Untouched | 100% byte-for-byte identical | **PASS** |

- *Metrics Saved:* [`results/metrics/preprocessing/preprocessing_validation.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/preprocessing/preprocessing_validation.csv)
- *Feature Matrices Saved:*
  - [`data/processed/train_features.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/train_features.csv)
  - [`data/processed/validation_features.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/validation_features.csv)
  - [`data/processed/test_features.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/test_features.csv)
- *Metadata Matrices Saved:*
  - [`data/processed/train_metadata.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/train_metadata.csv)
  - [`data/processed/validation_metadata.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/validation_metadata.csv)
  - [`data/processed/test_metadata.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/test_metadata.csv)

---

## 7. Technical Limitations of the Preprocessed Dataset
1. **Unbalanced Chiller Representation in Later Partitions:** Chiller-2 was taken offline after March 15; therefore, Chiller-2 is present only in the training partition and has 0 records in validation and test partitions.
2. **Training Typo Inclusion vs Robust PCA:** The training partition contains isolated manual logging errors (e.g. Row 297: Liquid Temp Comp 1 = 631.1 °C in Chiller-3). In Phase 4, we must evaluate how these extreme points influence component loadings and study the explained variance distribution.

