# Technical Preprocessing Notes and Anti-Data-Leakage Record

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Raw Source Dataset:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls`  
**Reference Document:** Phase 3 Implementation Artifacts  

---

## 1. Dataset Characteristics & Modeling Scope

- **Raw Dimensions:** Exactly 525 rows $\times$ 60 columns from sheet `Log_Chiller_South`.
- **Header Alignment:** Row index 2 (Excel row 3) correctly extracts True sensor names without column truncation.
- **Operating Scope Filter:** Filtered for steady-state dual-compressor operation where both `Motor Current Amps Comp 1` > 0 and `Motor Current Amps Comp 2` > 0.
  - Total dual-compressor records: **$N = 517$ observations**.
  - Excluded records: 5 single-compressor observations (representing scheduled partial shutdown) and 3 empty/header rows.
- **Multi-Chiller Fleet Pooling:** Observations pooled across 5 industrial chiller units:
  - `Chiller - 1`: 96 observations
  - `Chiller - 2`: 82 observations
  - `Chiller - 3`: 120 observations
  - `Chiller - 4`: 131 observations
  - `Chiller - 5`: 88 observations
- **Sample-to-Feature Ratio ($N/p$):** Pooling yields $N/p = 517 / 28 \approx 18.5$, providing robust multivariate covariance estimation compared to single-chiller modeling ($N \approx 100, N/p \approx 3.5$).

---

## 2. Feature Boundary & Exclusions

Exactly **28 core thermodynamic variables** were selected for the PCA input matrix:
- **21 Temperature Variables:**
  1. `Evaporator Water In Temperature Deg C`
  2. `Evaporator Water Out Temperature Deg C`
  3. `Condenser Water In Temperature Deg C`
  4. `Condenser Water Out Temperature Deg C`
  5. `Discharge Temperature Deg C Comp 1`
  6. `Discharge Temperature Deg C Comp 2`
  7. `Suction Temperature Deg C`
  8. `Chiler Lift Temperature Deg C Comp 1`
  9. `Chiler Lift Temperature Deg C Comp 2`
  10. `Suction Saturation Temperature Deg C`
  11. `Suction Superheat deg C`
  12. `Evaporator Approach` *(Deg C - Inferred)*
  13. `Discharge Saturation Temperature Deg C Comp 1`
  14. `Discharge Saturation Temperature Deg C Comp 2`
  15. `Discharge Superheat Deg C Comp 1`
  16. `Discharge Superheat Deg C Comp 2`
  17. `Condenser Approach Comp 1` *(Deg C - Inferred)*
  18. `Condenser Approach Comp 2` *(Deg C - Inferred)*
  19. `Liquid Temperature Deg C Comp 1`
  20. `Liquid Temperature Deg C Comp 2`
  21. `Oil Temperature Deg C`
- **7 Refrigerant Pressure Variables:**
  22. `Discharge Pressure Kg /cm 2 Comp 1`
  23. `Discharge Pressure Kg /cm 2 Comp 2`
  24. `Suction Pressure Kg /cm 2`
  25. `Chiller Lift Pressure Kg /cm 2 Comp 1`
  26. `Chiller Lift Pressure Kg /cm 2 Comp 2`
  27. `EXV Differential Pressure Kg /cm 2 Comp 1`
  28. `EXV Differential Pressure Kg /cm 2 Comp 2`

### Strict Non-Thermodynamic Exclusions:
- **Identifier & Timestamps:** `Sl.No`, `CHILLER TAG`, `Date`, `Time`, `Timestamp` (Retained in metadata files only).
- **Electrical Variables:** Motor currents, running loads, active power, power factors, energy meters, frequency (Retained in metadata for operational context; excluded from PCA).
- **Water Loop Pump Pressures:** Evaporator pump in/out and condenser pump in/out pressures (4 variables) excluded to prevent hydraulic variance from corrupting refrigerant thermodynamic modeling.

---

## 3. Data Quality & Pre-Imputation Findings

- **Missing Values:** Exactly 0.0% missing cells across all 28 features in the raw dataset ($100\%$ completeness).
- **Duplicate Rows:** Exactly 0 duplicate rows detected across the entire 60-column matrix.
- **Timestamp Concurrency:** Multiple chillers logging at identical timestamps represent synchronized multi-machine plant operations, not duplicate measurements.
- **Zero-Variance Features:** Zero constant features detected. `Suction Pressure Kg /cm 2` has low standard deviation ($\sigma = 0.081\text{ kg/cm}^2$), reflecting tight thermostatic expansion valve regulation.

---

## 4. Chronological Splitting Protocol (Zero Data Leakage)

To prevent future temporal leakage into past estimates, observations were sorted forward in time (`2026-03-11 00:00:00` to `2026-03-20 08:00:00`) and partitioned based on **unique plant timestamp hours** ($N_{\text{hours}} = 222$):

| Split Partition | Unique Hours | Start Timestamp | End Timestamp | Observation Count ($N$) | Percentage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 155 | 2026-03-11 00:00:00 | 2026-03-17 10:00:00 | **358** | **69.25%** |
| **Validation Set** | 33 | 2026-03-17 11:00:00 | 2026-03-18 19:00:00 | **92** | **17.79%** |
| **Testing Set** | 34 | 2026-03-18 20:00:00 | 2026-03-20 08:00:00 | **67** | **12.96%** |
| **Total Fleet** | **222** | **2026-03-11 00:00:00** | **2026-03-20 08:00:00** | **517** | **100.0%** |

### Leakage Prevention Guarantees:
1. **Zero Temporal Overlap:** Overlap between Train/Val = 0, Val/Test = 0, Train/Test = 0.
2. **Synchronized Integrity:** All concurrent chillers operating at a given timestamp are assigned to the exact same temporal split.
3. **No Random Shuffling:** Shuffling was strictly prohibited to preserve time-series autoregressive continuity.

---

## 5. Transformer Fitting & Standardization Protocol

- **Imputation:** `SimpleImputer(strategy='median')` was fitted strictly on $X_{\text{train}}$ only. Saved to `data/processed/imputer.joblib`.
- **Standardization:** `StandardScaler()` was fitted strictly on $X_{\text{train}}$ only. Saved to `data/processed/scaler.joblib`.
  - Training mean: $\mu_{\text{train}} = 0$ (maximum absolute value $< 4.63 \times 10^{-15}$).
  - Training standard deviation: $\sigma_{\text{train}} = 1.000000$ across all 28 features.
- **Validation and Test Transformation:**
  $$Z_{\text{val}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad Z_{\text{test}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  Neither imputer nor scaler was refitted or updated on validation or test observations.
- **Verification:** All 18 automated validation criteria in `results/metrics/preprocessing/preprocessing_validation.csv` scored **PASS**.

