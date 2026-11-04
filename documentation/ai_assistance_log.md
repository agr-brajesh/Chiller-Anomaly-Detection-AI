# AI Assistance and Academic Integrity Audit Log

**Student Register Number:** 24BCE0551  
**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Assigned Method:** PCA-Based Multivariate Detector  
**Institutional Policy Compliance:** AI-generated content < 10%, Similarity < 5%. Submissions must represent original independent work.

---

## Log Entries

### Entry 1: Phase 1 — Project Setup and Dataset Understanding
- **Date:** 2026-11-04
- **Task Performed:** Initial project setup, directory creation, dependency validation, comparison of dataset copies (SHA-256), inspection of all worksheets in raw Excel workbook, identification of schema, sampling frequency, sensor categories, and initial data-quality anomalies.
- **Nature of Assistance:** Technical mentoring, command execution for inspection, structuring the project workspace, and developing the automated diagnostic script `src/data_inspection.py`.
- **Code/Explanations Suggested:**
  - Setup script for project directory hierarchy according to project specifications.
  - Python script (`src/data_inspection.py`) implementing:
    - Package dependency checks (`xlrd`, `pandas`, `numpy`, `scipy`, `sklearn`, `matplotlib`, `seaborn`).
    - SHA-256 hash comparison between root and `data/raw/official_dataset/` copy (`7c3e80d5...`, 100% byte match).
    - Worksheet discovery (`1` visible sheet: `Log_Chiller_South`).
    - Header offset alignment (Row index 2 / Excel row 3) and column categorization.
  - Categorization of 60 raw variables: 4 identification/timestamp, 21 temperature, 7 refrigerant pressure, 13 compressor electrical, 3 actuators/controls, 2 run hours, and 10 auxiliary plant equipment.
  - Identification of reverse chronological sorting and multi-chiller logging (Chiller-1 through 5).
  - Diagnostic scan identifying industrial manual logging typos (e.g. 10x decimal errors in motor current and liquid temperatures).
- **Student Independent Work & Verification:**
  - Confirmed official dataset location at `data/raw/official_dataset/Analyse Chillers_AI Course.xls`.
  - Verified that root and raw files are identical via SHA-256 and confirmed neither file will be modified or deleted.
  - Confirmed exactly 1 worksheet (`Log_Chiller_South`), 522 observational rows, and 60 variables.
  - Reviewed the 21 temperature variables and 7 refrigerant pressure variables directly targeting the assigned research question.
  - Confirmed 0 null/NaN values across the entire matrix.
  - Verified reverse chronological order and planned chronological sorting for time-series modeling.
  - Validated all 5 Phase 1 deliverables in `documentation/`.
  - Updated `.gitignore` to ensure raw datasets are excluded from Git commits.

### Entry 2: Phase 1 Decisions & Feature Scope Verification
- **Date:** 2026-11-04
- **Task Performed:** Formalized and documented student's three modeling decisions; programmatically verified the 517 dual-compressor observations and the 28 core thermodynamic variables; investigated cross-machine sensor comparability; simulated chronological partitioning based on unique timestamps.
- **Nature of Assistance:** Technical mentoring, statistical advice regarding sample-to-feature ratio ($N/p$), creating verification script `src/verify_modeling_scope.py`, and updating decision records.
- **Code/Explanations Suggested:**
  - Mathematical justification for pooling ($N = 517, p = 28 \implies N/p \approx 18.5$ vs $N \approx 125 \implies N/p \approx 4.5$).
  - Python script `src/verify_modeling_scope.py` to filter dual-compressor rows (`Comp 1 > 0` and `Comp 2 > 0`), check missing values across 28 features, compare means/medians across Chillers 1-5, and simulate timestamp-based temporal splitting.
  - Documented approach temperatures as "Deg C (Inferred)".
- **Student Independent Decisions & Approvals:**
  - Approved Decision 1: Pooled fleet modeling under dual-compressor operation ($N = 517$), retaining `CHILLER TAG` for diagnostic tracking while excluding it from PCA input features.
  - Approved Decision 2: Restricting PCA input space strictly to the 28 core thermodynamic variables (21 temperatures + 7 refrigerant pressures) to address the research question, while retaining electrical/load variables as auxiliary diagnostics.
  - Approved Decision 3: Interpreting `Evaporator Approach` and `Condenser Approach Comp 1/2` as °C, documented as inferred units.
