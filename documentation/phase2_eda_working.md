# Phase 2 Working Document: Data Quality Assessment and Exploratory Data Analysis (EDA)

**Document Purpose:** Factual engineering notes recording empirical results from Phase 2 exploratory data analysis. Contains measured data-quality metrics, statistical summaries, correlation tables, and technical observations.

---

## 1. Activity A: Missing-Value Findings
- **Raw Dataset Scope:**
  - File: `data/raw/official_dataset/Analyse Chillers_AI Course.xls` (Sheet: `Log_Chiller_South`)
  - Dimensions: 522 rows $\times$ 60 columns (31,320 total data cells)
  - Total Missing Cells: **0**
  - Dataset Completeness: **100.0%**
  - Rows with $\ge 1$ Missing Value: **0** (0.0%)
- **Dual-Compressor Filtered Scope ($N = 517$):**
  - Dimensions: 517 rows $\times$ 28 core thermodynamic features (14,476 total feature cells)
  - Total Missing Cells: **0**
  - Feature Completeness: **100.0%**
- **Metric Table Generated:** [`results/metrics/eda/missing_values_summary.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/missing_values_summary.csv)
- **Visualization Generated:** [`results/figures/eda/missing_values_analysis.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/missing_values_analysis.png)
- **Technical Note for Phase 3 Preprocessing:** Because missing values are 0 across all relevant operational variables, no median or mean imputation is empirically required for normal data. However, an imputation step should be maintained in the preprocessing pipeline as a defensive mechanism for streaming/test data.

---

## 2. Activity B: Duplicate Analysis Findings
- **Exact Duplicate Rows:**
  - Raw Fleet ($N = 522$): **0 exact duplicate rows** (excluding serial index `Sl.No`).
  - Dual-Compressor Fleet ($N = 517$): **0 exact duplicate rows**.
- **Column Name Uniqueness:**
  - All 60 raw column names are unique (0 duplicate column headers).
- **Timestamp Repetition & Concurrency:**
  - Repeated timestamps across the whole plant sheet: **295 occurrences** in the 517-row dual set.
  - Explanation: This is **not a data defect**. The plant logs 5 separate physical chillers (`Chiller - 1` to `Chiller - 5`) on the same sheet. Multiple chillers run concurrently in the same hour (e.g., 2 chillers running together during 123 hours; 3 chillers during 50 hours; 4 chillers during 24 hours).
  - Within any single chiller stream (e.g., examining Chiller-4 alone), timestamp duplicates = **0**.
- **Metric Table Generated:** [`results/metrics/eda/duplicate_analysis.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/duplicate_analysis.csv)

---

## 3. Activity C: Timestamp Continuity & Sampling Interval Analysis
- **Parsed Range:** `2026-03-11 00:00:00` to `2026-03-20 08:00:00` (Span: 224.0 calendar hours).
- **Raw File Direction:** Recorded in reverse chronological order (newest-first). Chronological sort applied for time-series analysis.
- **Unique Plant Sampling Intervals:**
  - Total unique plant hours: 222 hours.
  - Dominant / Modal Sampling Interval: **1 hour (0 days 01:00:00)** — represents 219 out of 221 transitions (99.1%).
  - Median Interval: **1 hour**.
  - Minimum Interval: **1 hour**.
  - Maximum Interval: **2 hours** (Plant-wide gaps of 2 hours occur only twice: 2026-03-14 07:00 to 09:00, and 2026-03-18 19:00 to 21:00).
- **Per-Chiller Stream Continuity:**
  - `Chiller - 1` (96 rows): Span = 224.0 hr | Most frequent = 1 hr | 4 irregular intervals (longest gap: 4 days 17 hr when unit was switched off).
  - `Chiller - 2` (82 rows): Span = 104.0 hr | Most frequent = 1 hr | 4 irregular intervals (longest gap: 17 hr).
  - `Chiller - 3` (120 rows): Span = 178.0 hr | Most frequent = 1 hr | 5 irregular intervals (longest gap: 22 hr).
  - `Chiller - 4` (131 rows): Span = 200.0 hr | Most frequent = 1 hr | 7 irregular intervals (longest gap: 5 hr).
  - `Chiller - 5` (88 rows): Span = 221.0 hr | Most frequent = 1 hr | 13 irregular intervals (longest gap: 16 hr).
- **Metric Tables Generated:**
  - [`results/metrics/eda/timestamp_continuity_summary.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/timestamp_continuity_summary.csv)
  - [`results/metrics/eda/irregular_timestamp_intervals.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/irregular_timestamp_intervals.csv)
- **Visualization Generated:** [`results/figures/eda/timestamp_continuity.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/timestamp_continuity.png)

---

## 4. Activity D: Descriptive Statistical Findings
Summary of central tendency, spread, and quartiles across the 28 core thermodynamic variables ($N = 517$ dual-compressor observations):

### Key Temperatures (°C)
* **Evaporator Water Out:** Mean = $7.62\text{ }^\circ\text{C}$, Median = $7.60\text{ }^\circ\text{C}$, Std = $0.62\text{ }^\circ\text{C}$, $\text{IQR} = 0.80\text{ }^\circ\text{C}$ [Min = $6.5\text{ }^\circ\text{C}$, Max = $9.2\text{ }^\circ\text{C}$]. Shows stable tight regulation around building setpoint.
* **Evaporator Water In:** Mean = $11.02\text{ }^\circ\text{C}$, Median = $10.90\text{ }^\circ\text{C}$, Std = $4.13\text{ }^\circ\text{C}$, $\text{IQR} = 1.30\text{ }^\circ\text{C}$ [Min = $8.4\text{ }^\circ\text{C}$, Max = $24.8\text{ }^\circ\text{C}$].
* **Condenser Water In:** Mean = $29.57\text{ }^\circ\text{C}$, Median = $29.50\text{ }^\circ\text{C}$, Std = $1.57\text{ }^\circ\text{C}$, $\text{IQR} = 1.20\text{ }^\circ\text{C}$ [Min = $27.1\text{ }^\circ\text{C}$, Max = $59.0\text{ }^\circ\text{C}$].
* **Condenser Water Out:** Mean = $31.85\text{ }^\circ\text{C}$, Median = $32.00\text{ }^\circ\text{C}$, Std = $2.84\text{ }^\circ\text{C}$, $\text{IQR} = 1.50\text{ }^\circ\text{C}$ [Min = $3.0\text{ }^\circ\text{C}$, Max = $35.4\text{ }^\circ\text{C}$] *(Min of 3.0 °C reflects Row 388 typo)*.
* **Discharge Temp Comp 1:** Mean = $45.69\text{ }^\circ\text{C}$, Median = $45.30\text{ }^\circ\text{C}$, Std = $8.29\text{ }^\circ\text{C}$, $\text{IQR} = 3.60\text{ }^\circ\text{C}$ [Min = $35.6\text{ }^\circ\text{C}$, Max = $98.0\text{ }^\circ\text{C}$].
* **Suction Temp:** Mean = $5.50\text{ }^\circ\text{C}$, Median = $5.40\text{ }^\circ\text{C}$, Std = $0.85\text{ }^\circ\text{C}$, $\text{IQR} = 1.20\text{ }^\circ\text{C}$ [Min = $3.8\text{ }^\circ\text{C}$, Max = $8.5\text{ }^\circ\text{C}$].
* **Oil Temp:** Mean = $48.33\text{ }^\circ\text{C}$, Median = $48.20\text{ }^\circ\text{C}$, Std = $2.69\text{ }^\circ\text{C}$, $\text{IQR} = 4.30\text{ }^\circ\text{C}$ [Min = $41.1\text{ }^\circ\text{C}$, Max = $56.0\text{ }^\circ\text{C}$].

### Key Refrigerant Pressures (kg/cm²)
* **Suction Pressure:** Mean = $2.39\text{ kg/cm}^2$, Median = $2.40\text{ kg/cm}^2$, Std = $0.098\text{ kg/cm}^2$, $\text{IQR} = 0.10\text{ kg/cm}^2$ [Min = $2.1\text{ kg/cm}^2$, Max = $2.7\text{ kg/cm}^2$].
* **Discharge Pressure Comp 1:** Mean = $7.98\text{ kg/cm}^2$, Median = $8.00\text{ kg/cm}^2$, Std = $1.78\text{ kg/cm}^2$, $\text{IQR} = 0.40\text{ kg/cm}^2$ [Min = $6.9\text{ kg/cm}^2$, Max = $47.9\text{ kg/cm}^2$] *(Max 47.9 reflects Row 213 typo)*.
* **Discharge Pressure Comp 2:** Mean = $8.13\text{ kg/cm}^2$, Median = $7.90\text{ kg/cm}^2$, Std = $4.85\text{ kg/cm}^2$, $\text{IQR} = 0.40\text{ kg/cm}^2$ [Min = $7.0\text{ kg/cm}^2$, Max = $88.0\text{ kg/cm}^2$] *(Max 88.0 reflects Row 213 typo)*.
* **Chiller Lift Pressure Comp 1:** Mean = $5.52\text{ kg/cm}^2$, Median = $5.60\text{ kg/cm}^2$, Std = $0.44\text{ kg/cm}^2$, $\text{IQR} = 0.50\text{ kg/cm}^2$ [Min = $4.6\text{ kg/cm}^2$, Max = $6.8\text{ kg/cm}^2$].
* **EXV Differential Pressure Comp 1:** Mean = $4.32\text{ kg/cm}^2$, Median = $4.30\text{ kg/cm}^2$, Std = $0.36\text{ kg/cm}^2$, $\text{IQR} = 0.40\text{ kg/cm}^2$ [Min = $3.4\text{ kg/cm}^2$, Max = $6.7\text{ kg/cm}^2$].
- **Metric Table Generated:** [`results/metrics/eda/descriptive_statistics.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/descriptive_statistics.csv)
- **Visualization Generated:** [`results/figures/eda/sensor_distributions.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/sensor_distributions.png)

---

## 5. Activity E: Zero-Variance & Outlier Investigation
- **Zero-Variance Features:**
  - Exactly **0 features** have zero variance. Every single feature in the 28 core thermodynamic matrix exhibits positive variation.
- **Near-Zero Variance Analysis:**
  - Feature with lowest empirical variance: `Suction Pressure Kg /cm 2` ($\text{Variance} = 0.0096$, $\text{Std} = 0.098\text{ kg/cm}^2$).
  - Reason: Industrial chillers tightly regulate low-side evaporating pressure near 2.3–2.5 kg/cm² to maintain stable boiling of refrigerant without freezing the evaporator tubes. This low variance is normal physical regulation, not a sensor freeze defect.
- **Gross Data-Entry Typographical Outliers Identified:**
  1. *Motor Current Comp 1 (Row 460 & 475):* `1350.0 A` and `2196.0 A` (normal is ~`135 A` and `220 A`; classic $10\times$ decimal-shift typos).
  2. *Motor Current Comp 2 (Row 195):* `2106.0 A` (normal is ~`210 A`).
  3. *Liquid Temp Comp 1 (Row 297 & 441):* `631.1 °C` and `329.0 °C` (normal is ~`30 °C`).
  4. *Chiller Lift Temp Comp 2 (Row 391):* `415.0 °C` (normal is ~`40 °C`).
  5. *Condenser Water Out Temp (Row 388):* `3.0 °C` (inlet is $28.5\text{ }^\circ\text{C}$; water cannot exit cooling tower loop at $3.0\text{ }^\circ\text{C}$; likely a typo for $30.0\text{ }^\circ\text{C}$).
  6. *Discharge Pressure Comp 1 & 2 (Row 213 & 416):* `47.9 kg/cm²`, `88.0 kg/cm²`, and `82.0 kg/cm²` (normal is ~`8.0 kg/cm²`).
- **Metric Tables Generated:**
  - [`results/metrics/eda/zero_variance_features.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/zero_variance_features.csv)
  - [`results/metrics/eda/sensor_quality_summary.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/sensor_quality_summary.csv)
- **Visualization Generated:** [`results/figures/eda/zero_variance_analysis.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/zero_variance_analysis.png)

---

## 6. Activity F: Temperature-Pressure Correlation Analysis Findings

### Strongest Positive Correlations ($r > +0.70$)
1. **Discharge Temperature Comp 1 $\leftrightarrow$ Discharge Pressure Comp 1:**
   - Pearson $r = \mathbf{+0.9578}$, Spearman $\rho = \mathbf{+0.7641}$
   - *Physical Meaning:* Direct manifestation of the vapor compression thermodynamic cycle. As refrigerant gas is compressed to high condensing pressures, discharge temperature rises proportionally.
2. **Discharge Saturation Temp Comp 1 $\leftrightarrow$ Chiller Lift Pressure Comp 1:**
   - Pearson $r = \mathbf{+0.8541}$, Spearman $\rho = \mathbf{+0.8493}$
   - *Physical Meaning:* Saturation temperature directly tracks the saturation pressure curve.
3. **Chiller Lift Temp Comp 1 $\leftrightarrow$ Chiller Lift Pressure Comp 1:**
   - Pearson $r = \mathbf{+0.7685}$, Spearman $\rho = \mathbf{+0.6926}$
   - *Physical Meaning:* Temperature lift across the compressor is driven by the pressure lift work done on the gas.
4. **Liquid Temperature Comp 2 $\leftrightarrow$ Chiller Lift Pressure Comp 2:**
   - Pearson $r = \mathbf{+0.7372}$, Spearman $\rho = \mathbf{+0.7065}$
5. **Suction Saturation Temp $\leftrightarrow$ Suction Pressure:**
   - Pearson $r = \mathbf{+0.7043}$, Spearman $\rho = \mathbf{+0.6660}$
   - *Physical Meaning:* Direct boiling saturation curve of the refrigerant inside the evaporator.

### Strongest Negative Correlations ($r < -0.30$)
1. **Discharge Superheat Comp 2 $\leftrightarrow$ Suction Pressure:** Pearson $r = \mathbf{-0.4746}$, Spearman $\rho = \mathbf{-0.4965}$.
2. **Discharge Superheat Comp 1 $\leftrightarrow$ Suction Pressure:** Pearson $r = \mathbf{-0.3672}$, Spearman $\rho = \mathbf{-0.3537}$.
3. **Suction Superheat $\leftrightarrow$ Suction Pressure:** Pearson $r = \mathbf{-0.3521}$, Spearman $\rho = \mathbf{-0.2204}$.
4. **Evaporator Approach $\leftrightarrow$ Suction Pressure:** Pearson $r = \mathbf{-0.3235}$, Spearman $\rho = \mathbf{-0.6484}$.
   - *Physical Meaning:* When building cooling load increases, evaporating pressure rises, drawing denser refrigerant into the compressor and naturally reducing superheat.

### Answering the Primary Research Question:
The empirical correlation analysis confirms that **temperature and pressure variables exhibit strong, physically consistent multivariate coupling ($r$ up to $+0.96$)**. When the chiller is in normal health, observations occupy a low-dimensional subspace where temperature and pressure move together. When an anomaly occurs (sensor failure, refrigerant leak, fouling), this balance is broken, causing substantial projection error outside the PCA subspace.

- **Metric Tables Generated:**
  - [`results/metrics/eda/pearson_correlation_matrix.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/pearson_correlation_matrix.csv)
  - [`results/metrics/eda/spearman_correlation_matrix.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/spearman_correlation_matrix.csv)
  - [`results/metrics/eda/temperature_pressure_pairwise_correlations.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/temperature_pressure_pairwise_correlations.csv)
- **Visualizations Generated:**
  - [`results/figures/eda/temperature_pressure_heatmap.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/temperature_pressure_heatmap.png)
  - [`results/figures/eda/temperature_pressure_scatterplots.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/temperature_pressure_scatterplots.png)

---

## 7. Activity G: Time-Series Exploratory Findings
- Visualized Chiller - 4 (131 continuous observations) across both temperature and pressure panels.
- Temperatures exhibit diurnal cyclic behavior reflecting outdoor temperature shifts and building occupancy cooling demand.
- Pressures track temperatures synchronously, maintaining consistent differential lift.
- The isolated spike at 2026-03-16 11:00:00 (Row 213: Discharge Pressure = 47.9 kg/cm²) stands out prominently as a sharp point anomaly, visually demonstrating why PCA will be effective in detecting it.
- **Visualization Generated:** [`results/figures/eda/sensor_time_series_profiles.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/eda/sensor_time_series_profiles.png)

---

## 8. Summary of Preprocessing Considerations for Phase 3
1. **Chronological Sorting:** The dataset must be sorted chronologically ascending prior to train/val/test splitting.
2. **Partitioning by Unique Timestamps:** Partitions must be split by unique plant timestamps (e.g. 70% / 15% / 15%) so that concurrent observations from the same hour do not straddle partition boundaries.
3. **Training Data Cleansing / Quarantining:**
   - Standard PCA fits eigenvectors on the empirical covariance matrix.
   - Gross manual entry typos in the training partition (e.g. 631.1 °C, 1350 A, 47.9 kg/cm²) would artificially rotate principal components.
   - In Phase 3, we must ensure the training set is clean of extreme recording errors so the model learns true normal physical relationships.
4. **Feature Scaling (StandardScaler):** Since temperatures range from 3 °C to 98 °C and pressures range from 2 kg/cm² to 8 kg/cm², `StandardScaler` (zero mean, unit variance) fitted strictly on the training partition is mandatory.

