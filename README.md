# AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data

## Academic Metadata & Project Context
- **Student Register Number:** 24BCE0551
- **Track:** Multivariate and Time-Series Detection
- **Assigned Machine Learning Method:** Principal Component Analysis (PCA)-Based Multivariate Anomaly Detector
- **Primary Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*
- **Primary Evaluation Strategy:** PCA Reconstruction Error (Squared Prediction Error / SPE) & Synthetic Anomaly Testing (Spike, Drift, Correlation Breaking)
- **Assigned Preprocessing Protocol:** Imputation, StandardScaler, and Correlation Analysis (Strictly Training-Fitted)
- **Official Raw Dataset:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls` (Preserved Unmodified, SHA-256 Verified)

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
│   │   ├── train_features.csv                     # Standardized training feature matrix (358 x 28)
│   │   ├── validation_features.csv                # Standardized validation feature matrix (92 x 28)
│   │   ├── test_features.csv                      # Standardized testing feature matrix (67 x 28)
│   │   ├── train_metadata.csv                     # Training metadata (Timestamp, Chiller Tag, Loads)
│   │   ├── validation_metadata.csv                # Validation metadata
│   │   ├── test_metadata.csv                      # Testing metadata
│   │   ├── imputer.joblib                         # Fitted SimpleImputer (median, train-only)
│   │   ├── scaler.joblib                          # Fitted StandardScaler (train-only)
│   │   └── pca_model.joblib                       # Fitted PCA Detector (k = 12 components)
│   └── synthetic/                                 # Dedicated directory for Phase 6 synthetic copies
│
├── notebooks/
│   └── Chiller_PCA_Anomaly_Detection.ipynb        # Master Jupyter Notebook (Phases 1 to 4 integrated)
│
├── src/
│   ├── data_inspection.py                         # Phase 1: Dataset profiling & workbook inspection
│   ├── eda_analysis.py                            # Phase 2: EDA, quality checks & correlation matrix
│   ├── preprocessing.py                           # Phase 3: Anti-leakage chronological split & scaling
│   ├── pca_detector.py                            # Phase 4: PCA training, loadings & variance analysis
│   ├── anomaly_detection.py                       # Phase 5: SPE calculation & threshold determination
│   └── evaluation.py                              # Phase 6 & 7: Synthetic experiments & validation
│
├── results/
│   ├── figures/
│   │   ├── eda/                                   # 7 Phase 2 EDA figures (300 DPI)
│   │   │   ├── missing_values_analysis.png
│   │   │   ├── temperature_pressure_heatmap.png
│   │   │   ├── sensor_distributions.png
│   │   │   ├── sensor_time_series_profiles.png
│   │   │   ├── timestamp_continuity.png
│   │   │   ├── zero_variance_analysis.png
│   │   │   └── temperature_pressure_scatterplots.png
│   │   ├── preprocessing/                         # 2 Phase 3 Preprocessing figures (300 DPI)
│   │   │   ├── train_validation_test_distribution.png
│   │   │   └── scaling_comparison.png
│   │   └── pca/                                   # 6 Phase 4 PCA figures (300 DPI)
│   │       ├── scree_plot.png
│   │       ├── cumulative_explained_variance.png
│   │       ├── component_loadings.png
│   │       ├── component_contribution.png
│   │       ├── training_component_scores.png
│   │       └── chiller_component_diagnostics.png
│   │
│   ├── metrics/
│   │   ├── eda/                                   # 10 Phase 2 EDA metric tables
│   │   │   ├── missing_values_summary.csv
│   │   │   ├── duplicate_analysis.csv
│   │   │   ├── timestamp_continuity_summary.csv
│   │   │   ├── irregular_timestamp_intervals.csv
│   │   │   ├── descriptive_statistics.csv
│   │   │   ├── zero_variance_features.csv
│   │   │   ├── pearson_correlation_matrix.csv
│   │   │   ├── spearman_correlation_matrix.csv
│   │   │   ├── temperature_pressure_pairwise_correlations.csv
│   │   │   └── sensor_quality_summary.csv
│   │   ├── preprocessing/                         # 4 Phase 3 Preprocessing metric tables
│   │   │   ├── imputation_statistics.csv
│   │   │   ├── training_feature_variance.csv
│   │   │   ├── scaling_parameters.csv
│   │   │   └── preprocessing_validation.csv
│   │   └── pca_explained_variance.csv             # Phase 4 full 28-component variance spectrum
│   │   └── pca_component_loadings.csv             # Phase 4 28x12 eigenvector loadings matrix
│   │   └── pca_component_selection_comparison.csv # Phase 4 candidate selection trade-offs
│   │   └── pca_training_scores.csv                # Phase 4 training projection scores (358 x 12)
│   └── final_analysis/                            # Phase 7 & 8 final reports
│
├── documentation/
│   ├── ai_assistance_log.md                       # Transparent academic integrity assistance log
│   ├── dataset_profile_working.md                 # Phase 1 factual working profile
│   ├── initial_data_dictionary.csv                # Phase 1 60-variable classified dictionary
│   ├── phase1_file_inventory.csv                  # Phase 1 repository file inventory
│   ├── phase1_decision_log.md                     # Phase 1 approved engineering decisions
│   ├── phase2_eda_working.md                      # Phase 2 factual EDA working report
│   ├── phase3_preprocessing_working.md            # Phase 3 factual preprocessing working report
│   └── phase4_pca_working.md                      # Phase 4 factual PCA model working report
│
├── presentation/                                  # Phase 8 viva and slide deck materials
├── requirements.txt                               # Verified Python package dependencies
├── README.md                                      # Master comprehensive engineering documentation
└── .gitignore                                     # Git exclusions (data/raw/ and *.xls excluded)
```

---

## Phase 1 Summary: Official Dataset Discovery & Architecture

### 1. File Verification & Format
* **Raw File Path:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls`
* **File Size:** 665,088 bytes (649.5 KB)
* **SHA-256 Hash:** `7c3e80d5fdf3b97a4b833a6b0cd23fd81ba9b9ad860e6af42e19976a414c4914` (Matches root copy 100%).
* **Container Format:** Microsoft Excel 97–2004 Workbook (BIFF8 / OLE Compound File).
* **Worksheet Discovery:** Interrogation via `xlrd` confirms **exactly 1 visible worksheet** named `Log_Chiller_South` (0 hidden sheets).
* **Header Offset Structure:**
  * Excel Row 1 (Index 0): Title string (`Log - Chiller Plant`), remainder empty.
  * Excel Row 2 (Index 1): Completely blank (`NaN`).
  * Excel Row 3 (Index 2): **Actual 60 sensor headers** (`Sl.No`, `CHILLER TAG`, `Date`, `Time`, ...).
  * Excel Rows 4 to 525 (Index 3 to 524): **522 observational data rows**.
  * *Reading Configuration:* `pd.read_excel(path, sheet_name='Log_Chiller_South', header=2)`.

### 2. Fleet Composition & Temporal Inversion
* **Multi-Chiller Fleet:** The dataset aggregates logs from **5 distinct chillers** operating within the South Chiller Plant:
  * Chiller - 4: 131 observations
  * Chiller - 3: 120 observations
  * Chiller - 1: 96 observations
  * Chiller - 5: 89 observations
  * Chiller - 2: 86 observations
* **Temporal Sequence:** The raw file is logged in **reverse chronological order** (Row 0 = March 20, 08:00; Row 521 = March 11, 00:00). Observations must be sorted ascending before time-series splitting.
* **Operating Regime:**
  * Dual-Compressor Operation (`Comp 1 > 0` & `Comp 2 > 0`): **517 rows (99.04%)**.
  * Single-Compressor Operation: **5 rows (0.96%)** (Row 185, 194, 281, 335, 483).
  * Chiller Offline / Idle: 0 rows (all entries reflect active cooling).

### 3. Initial Industrial Logging Typo Discovery
Human manual log entry errors were identified in the raw sheet:
* **Motor Current Comp 1 (Row 460 & 475):** `1350.0 A` and `2196.0 A` (normal is ~`135 A` and `220 A`; classic $10\times$ decimal-shift slips).
* **Motor Current Comp 2 (Row 195):** `2106.0 A` (normal is ~`210 A`).
* **Liquid Temp Comp 1 (Row 297 & 441):** `631.1 °C` and `329.0 °C` (normal is ~`30 °C`).
* **Condenser Water Out Temp (Row 388):** `3.0 °C` (inlet is $28.5\text{ }^\circ\text{C}$; water cannot cool to $3.0\text{ }^\circ\text{C}$ in an evaporative cooling tower; typo for $30.0\text{ }^\circ\text{C}$).
* **Discharge Pressure Comp 1/2 (Row 213 & 416):** Pressures of `47.9 kg/cm²`, `88.0 kg/cm²`, and `82.0 kg/cm²` (normal is ~`8.0 kg/cm²`).

---

## Approved Modeling Scope & Feature Boundary

The modeling parameters approved for this project:

1. **Fleet Scope (Pooled Dual-Compressor Operation):**
   * Pool all 5 chillers under dual-compressor operation (**$N = 517$ observations**).
   * *Statistical Justification:* A single chiller offers only 120–131 observations. For 28 input variables, a single machine gives a sample-to-feature ratio of $N/p \approx 4.3\text{--}4.7$, which risks unstable covariance estimation. Pooling gives $N = 517$ ($N/p \approx 18.5$), yielding robust eigenvalue decomposition.
   * `CHILLER TAG` is retained for diagnostic tracking and error breakdowns, but **strictly excluded** from numerical PCA input features.
2. **Feature Boundary (28 Core Thermodynamic Features):**
   * The primary PCA detector is strictly restricted to the **28 core thermodynamic variables** (21 temperatures and 7 refrigerant pressures) directly addressing the research question.
   * Operational load and motor currents are excluded from PCA inputs to avoid confounding thermodynamic relationships, but retained in metadata.
3. **Approach Temperature Units:**
   * `Evaporator Approach`, `Condenser Approach Comp 1`, and `Condenser Approach Comp 2` are classified as temperature differences measured in °C, documented officially as **"Deg C (Inferred)"** until facility documentation independently confirms it.

---

## Complete 28 Core Thermodynamic Features List

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

## Phase 2 Summary: Data Quality Assessment & EDA Findings

### 1. Data Quality Audit
* **Completeness:** **100.0%** across all 60 columns and 522 rows (**0 missing cells**).
* **Exact Duplicates:** **0 duplicate rows** across the dataset.
* **Timestamp Repetitions:** 295 repeated timestamps occur in the sheet due to simultaneous operation of 2–4 chillers. Within each machine stream, there are **0 duplicate timestamps**.
* **Zero-Variance Features:** **0 zero-variance features**. All 28 features possess positive sample variance.
* **Sampling Regularity:** Dominant sampling interval is **1 hour (`0 days 01:00:00`)** across 222 unique plant hours.

### 2. Temperature-Pressure Correlation Analysis (Empirical Proof of Research Question)
The pairwise correlation analysis ($21 \times 7 = 147$ pairs) confirmed strong multivariate relationships:

* **Strongest Positive Couplings ($r > +0.70$):**
  1. `Discharge Temperature Comp 1` $\leftrightarrow$ `Discharge Pressure Comp 1`: **Pearson $r = \mathbf{+0.9578}$**, **Spearman $\rho = \mathbf{+0.7641}$**
  2. `Discharge Saturation Temp Comp 1` $\leftrightarrow$ `Chiller Lift Pressure Comp 1`: **Pearson $r = \mathbf{+0.8541}$**, **Spearman $\rho = \mathbf{+0.8493}$**
  3. `Chiler Lift Temperature Comp 1` $\leftrightarrow$ `Chiller Lift Pressure Comp 1`: **Pearson $r = \mathbf{+0.7685}$**, **Spearman $\rho = \mathbf{+0.6926}$**
  4. `Liquid Temperature Comp 2` $\leftrightarrow$ `Chiller Lift Pressure Comp 2`: **Pearson $r = \mathbf{+0.7372}$**, **Spearman $\rho = \mathbf{+0.7065}$**
  5. `Suction Saturation Temperature` $\leftrightarrow$ `Suction Pressure`: **Pearson $r = \mathbf{+0.7043}$**, **Spearman $\rho = \mathbf{+0.6660}$**
* **Strongest Negative Couplings:**
  1. `Discharge Superheat Comp 2` $\leftrightarrow$ `Suction Pressure`: **Pearson $r = \mathbf{-0.4746}$**, **Spearman $\rho = \mathbf{-0.4965}$**
  2. `Evaporator Approach` $\leftrightarrow$ `Suction Pressure`: **Pearson $r = \mathbf{-0.3235}$**, **Spearman $\rho = \mathbf{-0.6484}$**
* **Answer to Research Question:** **Yes.** Normal operating conditions exhibit strong linear and rank correlations governed by thermodynamic refrigeration laws. Because these sensors move together in a low-dimensional subspace, PCA can easily detect equipment faults, sensor drift, or line leaks that break this balance.

---

## Phase 3 Summary: Data Preprocessing & Anti-Data-Leakage Protocol

### 1. Chronological Partitioning by Unique Timestamps (70/15/15)
To prevent temporal leakage from concurrent chillers, splitting was computed strictly on the **222 unique hourly timestamps**:
* **Training Partition (First 70% of unique hours = 155 hours):**
  * Observations: **358 rows (69.25%)** | `2026-03-11 00:00:00` to `2026-03-17 10:00:00`
  * Chiller Representation: Chiller-1: 67, Chiller-2: 82, Chiller-3: 79, Chiller-4: 76, Chiller-5: 54
* **Validation Partition (Next 15% of unique hours = 33 hours):**
  * Observations: **92 rows (17.80%)** | `2026-03-17 11:00:00` to `2026-03-18 19:00:00`
  * Chiller Representation: Chiller-1: 17, Chiller-2: 0, Chiller-3: 25, Chiller-4: 33, Chiller-5: 17
* **Testing Partition (Final 15% of unique hours = 34 hours):**
  * Observations: **67 rows (12.96%)** | `2026-03-18 20:00:00` to `2026-03-20 08:00:00`
  * Chiller Representation: Chiller-1: 12, Chiller-2: 0, Chiller-3: 16, Chiller-4: 22, Chiller-5: 17
* **Temporal Integrity Check:** $\text{Train End} < \text{Val Start} < \text{Val End} < \text{Test Start}$. Strictly zero temporal overlap.

### 2. Transformers Fitted on Training Partition Only
* **Imputer:** `SimpleImputer(strategy="median")` fitted strictly on $X_{\text{train}}$ ($N = 358$). Saved to `data/processed/imputer.joblib`.
* **Scaler:** `StandardScaler` fitted strictly on $X_{\text{train}}$ ($Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$). Saved to `data/processed/scaler.joblib`.
* **Standardized Training Matrix Verification:** Mean max absolute error $= 4.62 \times 10^{-15} \approx 0.0$; Average std $= 1.0014 \approx 1.0$.

### 3. Preprocessing Validation Audit (18-Point Check)
All 18 validation criteria returned **PASS** (zero NaNs, zero Infs, strictly 28 features, strict chronological partitioning, raw source untouched).

---

## Phase 4 Summary: PCA Model Development

### 1. Mathematical Formulation
* **Sample Covariance Matrix:** $\mathbf{\Sigma} = \frac{1}{N_{\text{train}} - 1} X_{\text{train}}^T X_{\text{train}} \in \mathbb{R}^{28 \times 28}$
* **Eigenvalue Problem:** $\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$
* **Projection:** $\mathbf{z} = \mathbf{x} \mathbf{V}_k \in \mathbb{R}^k$
* **Reconstruction:** $\hat{\mathbf{x}} = \mathbf{z} \mathbf{V}_k^T = \mathbf{x} \mathbf{V}_k \mathbf{V}_k^T$
* **Residual Vector:** $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$
* **Squared Prediction Error (SPE):** $\text{SPE} = \|\mathbf{e}\|^2 = \sum_{j=1}^{28} (x_j - \hat{x}_j)^2$

### 2. Full Spectrum Variance & Component Selection Comparison

| Candidate Strategy | Retained Components ($k$) | Cumulative Variance (%) | Residual Dimensions ($28 - k$) | Residual Variance (%) | Technical Trade-off for Anomaly Detection |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Kaiser Rule ($\lambda \ge 1.0$)** | 8 | 72.33% | 20 | 27.67% | Discards >27% of normal physical variance, creating high baseline SPE noise on normal validation data. |
| **80% Target** | 11 | 82.50% | 17 | 17.50% | Acceptable compression, but slightly under-fits coupled secondary circuits. |
| **85% Target (SELECTED)** | **12** | **85.46%** | **16** | **14.54%** | **Optimal MSPC Balance:** Captures >85% of normal thermodynamic covariance while reserving 16 residual dimensions to catch anomalous decoupling. |
| **90% Target** | 14 | 90.85% | 14 | 9.15% | High fidelity; suitable if measurement noise is extremely small. |
| **95% Target** | 17 | 95.67% | 11 | 4.33% | Over-retention; higher components begin learning noise, reducing anomaly contrast in residual SPE. |

### 3. Component Loading Analysis (Physical Meaning of Subspaces)
* **PC1 (23.77% Variance): High-Side / Condenser Compression Work Subspace**
  * Dominant features: `Discharge Saturation Temp Comp 1` (+0.35), `Chiller Lift Pressure Comp 1` (+0.34), `Chiler Lift Temp Comp 1` (+0.30), `Liquid Temp Comp 2` (+0.30), `EXV Diff Pressure Comp 1` (+0.28).
  * Captures compression work: lift temperature, lift pressure, and condensing saturation temperature rise together.
* **PC2 (13.64% Variance): Low-Side / Evaporator Cooling Load Subspace**
  * Dominant features: `Suction Pressure` (+0.46), `Suction Saturation Temp` (+0.38), `Suction Temp` (+0.34), `Discharge Superheat Comp 2` (-0.33), `Condenser Approach Comp 2` (+0.31).
  * Captures cooling demand: higher evaporating load increases suction pressure and temperature while naturally lowering superheat.

---

## Technical Questions & Answers for Viva Preparation

1. **How many training observations were used to fit PCA?**
   * Exactly **358 observations** from the chronological training partition (March 11 to March 17, 10:00). Zero validation or test records were used.
2. **How many input features are in the PCA model?**
   * Exactly **28 features** (21 temperature variables and 7 refrigerant pressure variables).
3. **What variance does each retained component explain?**
   * PC1: 23.77%, PC2: 13.64%, PC3: 8.88%, PC4: 6.91%, PC5: 6.16%, PC6: 5.20%, PC7: 4.06%, PC8: 3.70%, PC9: 3.50%, PC10: 3.35%, PC11: 3.32%, PC12: 2.96%. Total: **85.46%**.
4. **Why was $k = 12$ chosen instead of retaining 95% variance?**
   * In reconstruction-error anomaly detection, retaining too many components ($k \ge 17$) allows anomalous patterns to be partially reconstructed by higher components, diminishing the error signal. Retaining $k = 12$ captures the true normal operating subspace while leaving 16 residual dimensions where decoupled faults will show high SPE.
5. **How will this PCA model be used in Phase 5?**
   * Validation and testing observations will be projected into score space ($\mathbf{z} = \mathbf{x}\mathbf{V}_{12}$) and inverse-reconstructed ($\hat{\mathbf{x}} = \mathbf{z}\mathbf{V}_{12}^T$). The reconstruction error $\text{SPE} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$ will be compared against a statistical threshold $\text{SPE}_{\text{threshold}}$ determined from normal validation data.
6. **Why is it essential that eigenvectors are orthogonal?**
   * Orthogonality ($\mathbf{v}_i^T \mathbf{v}_j = 0$ for $i \ne j$) ensures that the principal components are completely uncorrelated and non-redundant, allowing variance to be partitioned cleanly without double-counting.

---

## Execution Guide & How to Run

1. **Verify Environment Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Phase 1 Data Inspection:**
   ```bash
   python src/data_inspection.py
   ```
3. **Execute Phase 2 Exploratory Data Analysis & Figures:**
   ```bash
   python src/eda_analysis.py
   ```
4. **Execute Phase 3 Preprocessing & Anti-Leakage Partitioning:**
   ```bash
   python src/preprocessing.py
   ```
5. **Execute Phase 4 PCA Model Training & Diagnostics:**
   ```bash
   python src/pca_detector.py
   ```
6. **Open the Project Notebook:**
   ```bash
   jupyter lab notebooks/Chiller_PCA_Anomaly_Detection.ipynb
   ```

---

## Roadmap of Remaining Phases
* **Phase 5:** Reconstruction Error and Anomaly Detection (SPE calculation, validation thresholding, and test anomaly identification).
* **Phase 6:** Synthetic Anomaly Experiments (Spike, Drift, and Correlation-Breaking testing).
* **Phase 7:** Visualization and Result Interpretation (Final performance metrics, confusion matrices, and research question answer).
* **Phase 8:** Final Documentation, Report Preparation, and Viva Defense.
