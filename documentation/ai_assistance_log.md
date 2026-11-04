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

### Entry 3: Phase 2 — Data Quality Assessment and Exploratory Data Analysis (EDA)
- **Date:** 2026-11-04
- **Task Performed:** Executed complete Phase 2 analysis (Activities A through G): missing-value audit, duplicate record check, timestamp continuity & sampling interval analysis, descriptive statistics, zero/near-zero variance analysis, Pearson and Spearman correlation calculations between temperatures and pressures, and time-series profiling.
- **Nature of Assistance:** Developed modular analysis pipeline `src/eda_analysis.py`, executed headless visualization generation (300 DPI), calculated 10 metric CSVs, created working profile `documentation/phase2_eda_working.md`, and explained statistical meanings.
- **Code/Explanations Suggested:**
  - Missing-value audit: verified 100% completeness (0 missing cells).
  - Duplicate analysis: verified 0 exact duplicate rows, 0 duplicate columns; explained concurrency of chillers at identical timestamps.
  - Timestamp continuity: confirmed 1-hour modal interval across 222 unique plant hours.
  - Descriptive statistics: generated mean, median, std, min, max, Q25, Q50, Q75, and IQR for all features.
  - Zero-variance check: confirmed 0 zero-variance features; identified low-variance suction pressure as normal physical regulation.
  - Correlation analysis: discovered very strong thermodynamic correlations (e.g. Discharge Temp Comp 1 vs Discharge Pressure Comp 1: $r = +0.9578$, $\rho = +0.7641$) directly answering the research question.
  - Time-series plotting: separated temperature (°C) and pressure (kg/cm²) axes to prevent misleading scale distortions.
- **Student Independent Review & Next Steps:**
  - Review the 10 generated CSV tables and 7 analytical figures.
  - Evaluate the strong correlation findings supporting the primary research question.
  - Review data-quality implications (e.g. quarantining gross typos from training set) before Phase 3.


### Entry 4: Phase 3 — Data Preprocessing and Anti-Data-Leakage Protocol
- **Date:** 2026-11-04
- **Task Performed:** Implemented Phase 3 preprocessing pipeline (`src/preprocessing.py`) enforcing strict anti-data-leakage protocols: validated the 28 core thermodynamic features, executed chronological partitioning based on 222 unique plant timestamps (70% Train = 358 rows, 15% Val = 92 rows, 15% Test = 67 rows), fitted SimpleImputer(median) and StandardScaler strictly on the training partition, saved transformers and partitioned matrices, validated 18 anti-leakage audit criteria, generated figures, and updated the project notebook `notebooks/Chiller_PCA_Anomaly_Detection.ipynb`.
- **Nature of Assistance:** Technical mentoring, pipeline implementation, mathematical explanation of standardization and median imputation, preventing cross-chiller temporal leakage, generating metric tables, figures, working notes, and notebook structure.
- **Code/Explanations Suggested:**
  - Python script `src/preprocessing.py` implementing Activities A through F.
  - Mathematical definition of standardization: $Z = (X - \mu_{\text{train}}) / \sigma_{\text{train}}$.
  - Explanation of anti-leakage protocol: why concurrent chiller records at the same timestamp must be partitioned by unique timestamp rather than arbitrary row slicing.
  - Script `src/build_notebook.py` to generate Phase 3 notebook sections.
  - Factual working profile `documentation/phase3_preprocessing_working.md`.
- **Student Independent Review & Decisions:**
  - Reviewed the 28 core features in `data/processed/selected_features.json`.
  - Confirmed chronological split sizes (Train = 358, Val = 92, Test = 67).
  - Verified 18-point validation audit in `results/metrics/preprocessing/preprocessing_validation.csv`.
  - Confirmed that PCA training and thresholding remain untouched until Phase 4 approval.


### Entry 5: Phase 4 — PCA Model Development
- **Date:** 2026-11-04
- **Task Performed:** Implemented Phase 4 PCA detector training (`src/pca_detector.py`) strictly on the standardized training partition ($N_{\text{train}} = 358, p = 28$): computed full eigenvalue spectrum, evaluated candidate component selection rules (Kaiser, 80%, 85%, 90%, 95%), selected and justified $k = 12$ components (capturing 85.46% variance with 16 residual dimensions), trained final PCA model, analyzed eigenvector loadings, generated 6 diagnostic visualizations, saved model and configuration artifacts, and integrated Phase 4 into the project notebook.
- **Nature of Assistance:** Technical mentoring, mathematical explanation of eigenvalue decomposition and projection/reconstruction geometry, developing `src/pca_detector.py`, generating diagnostic figures, writing working document `documentation/phase4_pca_working.md`, and explaining thermodynamic interpretations of PC1 and PC2.
- **Code/Explanations Suggested:**
  - Python script `src/pca_detector.py` implementing Activities A through F.
  - Equations for sample covariance ($\mathbf{\Sigma} = \frac{1}{N-1}X^TX$), eigenvalue problem ($\mathbf{\Sigma}\mathbf{v} = \lambda\mathbf{v}$), EVR ($\lambda_i / 28$), projection ($\mathbf{z} = \mathbf{x}\mathbf{V}_k$), and reconstruction ($\hat{\mathbf{x}} = \mathbf{z}\mathbf{V}_k^T$).
  - Trade-off analysis for component selection in reconstruction-error-based anomaly detectors.
  - Identification of physical meaning in loadings: PC1 captures High-Side / Condenser Compression Lift, while PC2 captures Low-Side / Evaporator Cooling Demand.
- **Student Independent Review & Viva Preparation:**
  - Understand the mathematical difference between principal subspace and residual subspace.
  - Justify why $k = 12$ was selected over $k = 8$ or $k = 17$.
  - Explain how reconstruction error (SPE) will be derived from the 16 residual dimensions in Phase 5.


### Entry 6: Master Documentation & README Synchronization
- **Date:** 2026-11-04
- **Task Performed:** Synchronized and compiled the comprehensive repository documentation in `README.md`, integrating all findings, tables, mathematical formulations, 28-feature breakdowns, anti-leakage split metrics, PCA component loading interpretations, and viva Q&A from Phases 1 through 4 into a centralized master document.
- **Nature of Assistance:** Repository documentation structuring, technical writing, table formatting, and cross-linking to generated figures and metrics.
- **Student Verification:** Confirmed that `README.md` now contains the full end-to-end technical record of all project phases completed so far.


### Entry 7: Comprehensive Project Audit (Phases 1 to 4)
- **Date:** 2026-11-04
- **Task Performed:** Comprehensive technical, implementation, and documentation audit across Phases 1 through 4 prior to initiating Phase 5. Conducted read-only verification of dataset integrity (SHA-256), schema dimensions, 28-feature thermodynamic boundary, anti-leakage chronological splitting (Train: 358, Val: 92, Test: 67; 0 timestamp overlap), training-only fitting of `SimpleImputer` and `StandardScaler`, PCA mathematical orthogonality ($1.998 \times 10^{-15}$ error), invertibility (Train residual MSE $= 0.145390$), loading interpretations, 20 cross-phase consistency parameters, documentation completeness, and 10-mark rubric alignment.
- **Nature of Assistance:** Technical auditor, verification scripting (`src/audit_verifier.py`, `src/generate_cross_consistency_csv.py`), compiling formal audit reports (`phase1_to_phase4_comprehensive_audit.md`, `phase1_to_phase4_documentation_audit.md`, `phase1_to_phase4_cross_consistency.csv`, `phase1_to_phase4_correction_checklist.md`), and providing independent student report-writing guidance.
- **Student Independent Review & Decisions:**
  - Verified that all 20 cross-phase consistency parameters scored PASS with zero contradictions.
  - Confirmed that no models were retrained, no transformers refitted, and no raw data modified during the audit.
  - Confirmed strict test set isolation ($N = 67$) for Phase 5 threshold calibration.
  - Reviewed recommendations for independent student report writing and viva defense preparation to ensure < 10% AI-generated content and < 5% similarity compliance.
  - Evaluated final readiness decision: `READY TO PROCEED TO PHASE 5` pending student authorization.


### Entry 8: Phase 5 — Reconstruction Error and PCA-Based Anomaly Detection
- **Date:** 2026-11-04
- **Task Performed:** Implemented Phase 5 anomaly detection pipeline (`src/anomaly_detection_phase5.py`) using previously fitted artifacts (`pca_model.joblib`, `scaler.joblib`, `imputer.joblib`): performed projection and reconstruction in standardized feature space ($\hat{\mathbf{z}} = \mathbf{z}\mathbf{V}_k^T$), computed Squared Prediction Error ($\text{SPE} = \sum e_{ij}^2$) across all 28 features, evaluated validation threshold candidates (90%, 95%, 97.5%, 99%), discovered extreme tail distortion at the 99th percentile caused by a human logging typo ($322^\circ\text{C}$ in validation set), justified and recommended the 95th percentile threshold ($\tau = 9.2931$), applied the frozen threshold to the held-out test partition ($N_{\text{test}} = 67$), flagged 2 potential anomalies (2.99%), computed sensor-wise squared residual contributions (identifying that low-side suction and evaporator variables account for >73% of reconstruction error), analyzed chiller-specific distributions, generated 6 publication-quality figures, created 10 metric files, wrote `documentation/phase5_anomaly_detection_working.md`, and updated the project notebook.
- **Nature of Assistance:** Technical mentoring, pipeline implementation, statistical explanation of order statistics and threshold stability under small sample sizes, sensor contribution analysis, visualization generation, and working documentation.
- **Code/Explanations Suggested:**
  - Python script `src/anomaly_detection_phase5.py`.
  - Mathematical definition of SPE as sum of squared residuals: $\text{SPE}_i = \sum_{j=1}^{28} (z_{ij} - \hat{z}_{ij})^2$.
  - Analysis of order statistics in validation set ($N_{\text{val}} = 92$) and why the 95th percentile is far more robust to extreme logging typos than the 99th percentile.
  - Explanation of sensor-wise contribution formula: $\text{Contrib}_{ij} = e_{ij}^2 / \text{SPE}_i \times 100\%$.
  - Clarified why supervised metrics (Precision, Recall, ROC-AUC) must NOT be computed on real operational test data in the absence of verified ground truth.
- **Student Independent Review & Decisions:**
  - Review validation SPE statistics and the empirical cause of the 99th percentile distortion ($322^\circ\text{C}$ saturation typo).
  - Review and confirm recommended provisional threshold ($\tau = 9.2931$).
  - Evaluate the 2 flagged test observations (Chiller-5 at 02:00 on March 20 and Chiller-4 at 17:00 on March 19).
  - Understand physical sensor contributions (suction temperature and lift temperature deviations).
  - Confirm stop condition: wait for student authorization before proceeding to Phase 6 synthetic anomaly testing.


### Entry 9: Phase 6 — Synthetic Anomaly Experiment and Model Evaluation
- **Date:** 2026-11-04
- **Task Performed:** Implemented Phase 6 controlled synthetic perturbation experiments (`src/synthetic_experiments_phase6.py`) evaluating the frozen PCA detector ($\tau = 9.2931$) across three independent scenarios: Scenario A (Sudden Spikes, $\pm 4.0\sigma$, 3 single-observation events), Scenario B (Gradual Drift, linearly ramping from $+0.7\sigma$ to $+3.5\sigma$ over 5 hours in Chiller-1), and Scenario C (Temperature-Pressure Relationship Breaking, perturbing `Discharge Pressure Comp 1` by $-3.5\sigma$ over 3 hours while keeping correlated `Discharge Temp Comp 1` untouched, directly targeting the research question). Evaluated observation-level metrics (Precision, Recall, F1, Accuracy, FPR), confusion matrices, event-level detection rates (100% across all 3 scenarios), and detection delays. Generated 7 publication-quality figures, 8 metric CSVs, 6 synthetic datasets/ground-truth files, wrote `documentation/phase6_synthetic_experiment_working.md`, and updated the project notebook.
- **Nature of Assistance:** Technical mentoring, experimental design of standardized perturbation scenarios, pipeline implementation, explaining the mathematical difference between observation-level recall and event-level detection, generating high-resolution figures, and documenting research question verification.
- **Code/Explanations Suggested:**
  - Python script `src/synthetic_experiments_phase6.py`.
  - Explanation of standardized perturbation: $\Delta z = \pm 4.0$ corresponds directly to $4.0$ training standard deviations in physical units.
  - Explanation of baseline false positives: the 2 naturally occurring clean test excursions appear as $FP = 2$ under synthetic label $0$, resulting in net synthetic $FP = 0$.
  - Analysis of Scenario C proving that breaking the $r = +0.9578$ temperature-pressure physical equilibrium causes immediate projection into the 16 residual dimensions, triggering detection on the first affected observation.
- **Student Independent Review & Decisions:**
  - Review the injection manifest and verify within-chiller chronological continuity.
  - Review observation-level vs event-level performance trade-offs in gradual drift (20% observation recall vs 100% event detection).
  - Review empirical proof directly answering the assigned research question.
  - Confirm stop condition: hold progress until Phase 6 is reviewed and approved.


### Entry 10: Phase 7 — Final Visualization, Results Analysis and Interpretation
- **Date:** 2026-11-04
- **Task Performed:** Consolidated experimental results across Phases 1 through 6 (`src/final_results_phase7.py`); generated 12 publication-quality final visualizations (300 DPI) in `results/figures/final/`; compiled consolidated numerical comparison (`final_experiment_comparison.csv`) and research question evidence table (`research_question_evidence.csv`); produced visualization catalog (`phase7_visualization_index.md`) and technical analysis working notes (`phase7_results_analysis_working.md`); integrated Phase 7 into the master project notebook.
- **Nature of Assistance:** Technical mentoring, visualization formatting (300 DPI, academic typography, non-overlapping layouts), evidence synthesis, compiling cross-phase comparative metrics, and identifying empirical limitations.
- **Code/Explanations Suggested:**
  - Python script `src/final_results_phase7.py`.
  - Generation of the 12 final figures with clear epistemological distinctions between clean baseline, injected perturbations, and detector alarms.
  - Structure of the 7-claim Research Question Evidence Matrix categorizing claims as *Supported by evidence*, *Partially supported*, or *Cannot be determined*.
  - Identification of empirical limitations: small test sample size ($N_{\text{test}} = 67$), linear PCA assumptions, validation typo tail distortion, and absence of real maintenance work orders.
- **Student Independent Action & Review:**
  - Student reviews the 12 publication-quality figures and the consolidated experiment table.
  - Student uses the evidence table to formulate their own original academic conclusions and report sections.
  - Stop condition respected: no proceeding to Phase 8 without explicit student authorization.
