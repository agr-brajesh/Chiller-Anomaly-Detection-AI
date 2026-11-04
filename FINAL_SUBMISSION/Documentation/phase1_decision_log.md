# Phase 1 Decision Log & Technical Question Tracker (Approved)

**Student Register Number:** 24BCE0551  
**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Assigned Method:** PCA-Based Multivariate Detector  
**Phase:** Phase 1 — Dataset Understanding and Project Setup

---

## 1. What Was Inspected
1. **File Locations & Storage:**
   - Evaluated `Analyse Chillers_AI Course.xls` located in `./` and `data/raw/official_dataset/`.
   - Verified file sizes (665,088 bytes) and SHA-256 hashes (`7c3e80d5...`, identical match).
2. **Workbook Architecture & Worksheet Discovery:**
   - Low-level `xlrd` inspection across all workbook records (`wb.nsheets`, `sh.visibility`).
   - Detected 1 visible worksheet: `Log_Chiller_South`. Zero hidden sheets.
3. **Tabular Formatting & Header Alignment:**
   - Raw Excel row 1: Document title string.
   - Raw Excel row 2: Blank / NaN row.
   - Raw Excel row 3: 60 sensor headers.
   - Data rows: Rows 4 to 525 (522 observational records).
4. **Sensor Inventory & Operational State Mapping:**
   - 21 temperature variables (°C), 7 refrigerant pressure variables (kg/cm²), 4 auxiliary pump pressures (Bar), 13 electrical/power variables, 3 actuator variables, 2 run-hour counters, 10 auxiliary pump/fan variables.
5. **Data Quality & Timestamp Continuity:**
   - Zero missing values across the 522 $\times$ 60 matrix.
   - Reverse chronological sorting in raw sheet.
   - Manual data-entry typos identified (10x decimal errors in motor currents and extreme liquid temperatures).

---

## 2. Key Discoveries
- **Discovery 1 (Single Sheet Multi-Machine Fleet):** The dataset records 5 physical chillers (`Chiller - 1` through `Chiller - 5`) operating concurrently in the South Plant.
- **Discovery 2 (Reverse Chronological Order):** The raw logs are ordered newest-first (March 20 down to March 11). Chronological sorting is required before time-series partitioning.
- **Discovery 3 (Operating Regimes):** 99.04% (517/522 rows) represent dual-compressor operation. Only 5 rows represent single-compressor operation, and 0 rows represent offline periods.
- **Discovery 4 (Existing Gross Industrial Anomalies):** The raw records contain human logging typos ($10\times$ decimal errors in motor current: 1350 A, 2196 A, 2106 A; liquid temperatures: 631.1 °C, 329.0 °C; condenser water out: 3.0 °C).

---

## 3. Approved Modeling Decisions & Technical Justification

### Decision 1: Modeling Scope — Pooled Fleet under Dual-Compressor Operation (APPROVED)
* **Student Decision:** Pool observations across all 5 chillers under dual-compressor operation ($N = 517$ observations).
* **Justification:** A single chiller offers only 120–131 observations. Estimating a $28 \times 28$ covariance matrix with $N \approx 120$ yields a sample-to-feature ratio of $N/p \approx 4.3$ to $4.7$, which risks unstable covariance estimation. Pooling gives $N = 517$ ($N/p \approx 18.5$), providing substantially more robust eigenvector estimation.
* **Safeguards Applied:**
  1. `CHILLER TAG` is retained for diagnostic tracking and between-machine residual comparisons, but **strictly excluded** from PCA numerical inputs.
  2. Single-compressor records (5 rows: Row 185, 194, 281, 335, 483) are filtered out to ensure a homogeneous dual-compressor operating regime.
  3. Time-series partitioning will be performed on **unique plant timestamps**, ensuring all concurrent chiller records for any given hour fall into the same partition, preventing temporal data leakage.

### Decision 2: Feature Boundary — 28 Core Thermodynamic Variables (APPROVED)
* **Student Decision:** Restrict primary PCA input matrix strictly to the **28 core thermodynamic variables**:
  - **21 Temperature variables (°C)**
  - **7 Refrigerant pressure variables (kg/cm²)**
* **Justification:** Directly matches the assigned research question: *"Can anomalies be detected from correlated temperature and pressure variables?"* Operational variables (compressor load, motor current, pump currents) are excluded from PCA inputs to avoid confounding thermodynamic relationships, but retained as diagnostic reference variables for interpreting detected anomalies.

### Decision 3: Approach Temperature Units (APPROVED)
* **Student Decision:** Interpret `Evaporator Approach`, `Condenser Approach Comp 1`, and `Condenser Approach Comp 2` as temperature differences measured in °C.
* **Justification & Documentation:** Thermodynamically, approach is defined as $\Delta T = T_{\text{fluid}} - T_{\text{saturation}}$. Documented as **"Deg C (Inferred)"** in the data dictionary to distinguish physical domain inference from explicit sheet labeling.

---

## 4. Verification of the 28 Core Thermodynamic Features & Multi-Chiller Uniformity

The 28 variables were programmatically verified across all 517 dual-compressor observations:
* Missing values: **0** across all 28 features in all 517 rows.
* Multi-machine consistency:
  - `Evaporator Water Out Temp`: Means range from 7.37 °C to 7.78 °C across Chillers 1 to 5.
  - `Condenser Water In Temp`: Means range from 29.18 °C to 29.91 °C across Chillers 1 to 5.
  - `Suction Pressure`: Means range from 2.31 to 2.47 kg/cm² across Chillers 1 to 5.
  - `Discharge Pressure Comp 1`: Medians range from 7.8 to 8.2 kg/cm² across Chillers 1 to 5.
* Conclusion: The 5 machines share identical thermodynamic setpoints and comparable sensor distributions, supporting pooled multivariate modeling.
