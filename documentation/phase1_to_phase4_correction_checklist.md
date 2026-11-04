# Phase 1 to Phase 4 Technical Correction & Improvement Checklist

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Audit Date:** 2026-11-04  
**Audit Scope:** Prioritized review of technical implementation, anti-leakage compliance, documentation consistency, and viva preparation.

---

## Priority Classification Scheme

- **CRITICAL:** High-risk issues that compromise mathematical validity, cause data leakage, or violate instructor mandates. *Must be resolved before continuing.*
- **HIGH:** Technical or methodological items that impact evaluation rigor, marking rubric performance, or academic integrity compliance.
- **MEDIUM:** Documentation, traceability, or presentation improvements that enhance clarity and reproducibility.
- **LOW:** Minor organizational, code cleanup, or file hygiene items.

---

## 1. Summary of Identified Items

| ID | Priority | Category | Issue Description | Impact | Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **C-01** | **CRITICAL** | Data Leakage Check | Verification of training-only transformer fitting | Mathematical integrity of detector | **VERIFIED PASS (No Action Needed)** |
| **C-02** | **CRITICAL** | Model Restriction | Confinement strictly to assigned PCA method | Compliance with instructor track assignment | **VERIFIED PASS (No Action Needed)** |
| **H-01** | **HIGH** | Academic Integrity | AI Assistance Log update for Phase 1–4 audit | Mandatory compliance for <10% AI audit | **COMPLETED (Entry 7 Added)** |
| **H-02** | **HIGH** | Anti-Leakage Protocol | Strict isolation of test set ($N=67$) in Phase 5 | Prevention of threshold optimization leakage | **ACTION REQUIRED IN PHASE 5** |
| **H-03** | **HIGH** | Academic Originality | Student independent report drafting & viva prep | Compliance with <10% AI and <5% similarity | **STUDENT ACTION REQUIRED** |
| **M-01** | **MEDIUM** | Documentation Traceability | Consistent figure & metric file naming across docs | Eliminates minor naming ambiguities | **RESOLVED IN AUDIT MATRIX** |
| **M-02** | **MEDIUM** | Domain Justification | Explicit documentation of approach temperature units | Epistemological clarity in report | **VERIFIED PASS (Noted Inferred °C)** |
| **M-03** | **MEDIUM** | Data Quality Profiling | Explicit commentary on 3 irregular sampling intervals | Rigor in explaining plant logging behavior | **DOCUMENTED IN EDA SUMMARY** |
| **L-01** | **LOW** | Workspace Cleanliness | Organization of temporary verification scripts | Clean workspace for final repository export | **ORGANIZED (Scratch directory used)** |
| **L-02** | **LOW** | Version Control Hygiene | Update `.gitignore` for final submission archive | Prevents accidental upload of raw datasets | **VERIFIED PASS** |

---

## 2. Detailed Item Breakdown & Guidance

### CRITICAL Items (0 Active Blockers)

#### Item C-01: Training-Only Transformer and PCA Fitting
- **Evaluation:** Scaler, imputer, and PCA must be fitted solely on $X_{\text{train}}$ ($N = 358$).
- **Audit Findings:** Verified that `imputer.joblib`, `scaler.joblib`, and `pca_model.joblib` were trained exclusively on the 358 training observations. Training standardized feature means are zero ($< 4.63 \times 10^{-15}$), standard deviations are $1.000000$, and validation/test matrices were transformed using training statistics only. Zero temporal overlap was detected across splits.
- **Conclusion:** **CRITICAL PASS**. No corrective action required.

#### Item C-02: Strict Confinement to Assigned Algorithm
- **Evaluation:** Case study mandates PCA-based multivariate detection. No unauthorized deep learning or alternate algorithms may be implemented.
- **Audit Findings:** Verified that only linear PCA (`scikit-learn PCA(svd_solver='full')`) has been implemented, serialized, and evaluated.
- **Conclusion:** **CRITICAL PASS**. No corrective action required.

---

### HIGH Priority Items

#### Item H-01: Academic Integrity Audit Log Update
- **Issue:** The AI Assistance Log (`documentation/ai_assistance_log.md`) had recorded Entries 1 through 6. The comprehensive audit across Phases 1 to 4 must be formally logged.
- **Action Taken:** Appended Entry 7 documenting the read-only audit verification, script execution, file inventory review, and rubric alignment.
- **Status:** **RESOLVED**.

#### Item H-02: Test Set Isolation During Phase 5 Threshold Tuning
- **Requirement:** During Phase 5 (Reconstruction Error & Threshold Selection), anomaly thresholds (e.g., Q-statistic / SPE thresholds based on Jackson-Mudholkar or empirical percentiles) must be calibrated using the Training and Validation sets only.
- **Directive:** The test partition ($N = 67$) must remain completely held out until Phase 6 evaluation. Do NOT calculate threshold statistics on the test set.
- **Status:** **ENFORCED FOR PHASE 5 IMPLEMENTATION**.

#### Item H-03: Independent Report Writing & Viva Defense Preparation
- **Requirement:** Instructor policy mandates < 10% AI-generated content and < 5% similarity on final report submission.
- **Guidance for Student:**
  - Do not use AI to draft full report sections or copy-paste text blocks into the final submission document (PDF/DOCX).
  - Use the factual summaries, data tables, and figures from this workspace as empirical source material.
  - Write your own interpretations of the thermodynamic correlations ($r = 0.9578$), the scree plot trade-off ($k = 12$, 85.46% variance), and the physical meaning of PC1 and PC2.
  - Review the 10 Viva Q&As in `README.md` to prepare for oral questioning.
- **Status:** **STUDENT GUIDANCE PROVIDED**.

---

### MEDIUM Priority Items

#### Item M-01: Cross-Phase Figure and Metric Naming Consistency
- **Issue:** The EDA figure for missing values is named `missing_values_analysis.png`, the preprocessing split figure is named `train_validation_test_distribution.png`, and the preprocessing variance metric is named `training_feature_variance.csv`. Minor text references in working notes occasionally omitted suffixes (e.g. `_analysis` or `_distribution`).
- **Correction:** The documentation completeness matrix in `documentation/phase1_to_phase4_documentation_audit.md` explicitly lists both the expected and canonical filenames on disk. All files exist and match disk paths.
- **Status:** **RESOLVED IN AUDIT MATRIX**.

#### Item M-02: Epistemological Clarity for Inferred Units
- **Issue:** Excel headers for `Evaporator Approach`, `Condenser Approach Comp 1`, and `Condenser Approach Comp 2` omit physical units.
- **Audit Findings:** Verified that `initial_data_dictionary.csv`, `dataset_profile_working.md`, and `README.md` clearly label these variables as `Deg C (Inferred)`.
- **Status:** **VERIFIED PASS**.

#### Item M-03: Clarification of Plant Logging Gaps
- **Issue:** EDA detected 3 sampling intervals exceeding 1 hour (one 1.5-hour gap, two 2.0-hour gaps) in `results/metrics/eda/irregular_timestamp_intervals.csv`.
- **Guidance:** In the written report, clearly explain that these minor gaps reflect plant operational logging intervals rather than equipment downtime, demonstrating thorough data understanding.
- **Status:** **DOCUMENTED IN AUDIT FINDINGS**.

---

### LOW Priority Items

#### Item L-01: Organization of Verification Scripts
- **Action:** Read-only inspection scripts (`src/audit_verifier.py`, `src/generate_cross_consistency_csv.py`, `src/check_doc_paths.py`) and temporary files (`scratch/`, `audit_prompt.txt`) have been isolated to maintain a clean source repository.
- **Status:** **ORGANIZED**.

#### Item L-02: Version Control Hygiene
- **Action:** Verified that `.gitignore` properly excludes `.xls` raw datasets, `__pycache__`, `.ipynb_checkpoints`, and temporary system files, ensuring compliance with instructor requirement to avoid uploading the raw dataset to public repos.
- **Status:** **VERIFIED PASS**.

---

## 3. Final Conclusion & Phase 5 Readiness

With zero critical blockers, all high-priority compliance requirements satisfied, and all technical verifications passing with double-precision accuracy:

$$\mathbf{STATUS:\ READY\ TO\ PROCEED\ TO\ PHASE\ 5}$$

*(Pending student review and explicit authorization)*

