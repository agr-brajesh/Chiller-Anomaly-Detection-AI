# Comprehensive Viva Defense and Conceptual Learning Guide

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Purpose:** Technical mastery of mathematical, statistical, and engineering concepts for oral defense.  

---

## Topic A: Dataset Schema and Research Question

### Conceptual Foundation:
An industrial centrifugal chiller is a closed-loop thermodynamic system. Refrigerant (R134a) undergoes compression, condensation, expansion, and evaporation. Physical laws dictate that temperature and pressure in two-phase saturation regions are strictly coupled via the Clausius-Clapeyron equation and refrigerant vapor-pressure curves.

### Relevant Project File:
- [`documentation/dataset_profile_working.md`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/documentation/dataset_profile_working.md)
- [`results/metrics/eda/pearson_correlation_matrix.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/pearson_correlation_matrix.csv)

### Actual Project Findings:
- Dataset: 525 raw rows $\times$ 60 columns. Filtered for dual-compressor operation ($N = 517$).
- Evaluated 28 core thermodynamic variables (21 temperatures, 7 refrigerant pressures).
- Observed extremely strong collinearity: `Discharge Temperature Comp 1` vs `Discharge Pressure Comp 1` has $r = +0.9578$.

### Viva Question & Answer:
**Examiner:** *"Why did you focus exclusively on 28 temperature and pressure variables rather than using all 60 columns?"*  
**Defense Answer:** *"The assigned research question specifically asks whether anomalies can be detected from correlated temperature and pressure variables. The raw dataset contains non-thermodynamic variables such as motor current, power factor, run hours, and pump water pressures. Pump pressures reflect external water loop hydraulics rather than internal refrigerant cycle physics. By confining the input matrix strictly to the 28 core thermodynamic variables, we isolate the refrigerant phase equilibrium and prevent external electrical noise from corrupting the covariance matrix."*

### Self-Check Question:
*Can you explain why motor currents were excluded from the PCA matrix even though they correlate with load?*  
*(Answer: Motor currents are operational control actuators, not thermodynamic state variables. Retaining them in metadata allows post-detection operational context without distorting the thermodynamic covariance structure).*

---

## Topic B: Missing Values and Data Quality

### Conceptual Foundation:
In industrial telemetry, missingness can result from sensor disconnection (MCAR), operational outages (MAR), or physical clipping outside range limits (MNAR). Evaluating data completeness before training prevents premature imputation bias.

### Relevant Project File:
- [`documentation/preprocessing_notes.md`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/documentation/preprocessing_notes.md)
- [`results/metrics/eda/missing_values_summary.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/eda/missing_values_summary.csv)

### Actual Project Findings:
- Exactly 0.0% missing cells across all 28 features in the raw dataset (100% complete).
- Detected manual logging typos (e.g., motor current logged as 1350 A instead of 135 A; discharge saturation temperature logged as 322 °C instead of 32.2 °C).

### Viva Question & Answer:
**Examiner:** *"If there were zero missing values, why did you include a SimpleImputer in your preprocessing pipeline?"*  
**Defense Answer:** *"In production machine learning pipelines, maintaining an explicit imputer fitted strictly on training data is an essential engineering safeguard. While the historical batch had zero missing values, future streaming operational data from physical chiller sensors could encounter intermittent telemetry dropouts. Serializing the fitted median imputer ensures that any future missing value can be imputed using historical training medians without causing runtime pipeline failure or data leakage."*

---

## Topic C: Chronological Splitting vs. Random Splitting

### Conceptual Foundation:
In time-series and continuous plant monitoring, observations exhibit strong temporal autocorrelation. Randomly shuffling observations distributes past and future points across all partitions, allowing future operational states to leak into training statistics.

### Relevant Project File:
- [`data/processed/split_manifest.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/split_manifest.json)
- [`results/figures/preprocessing/train_validation_test_distribution.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/preprocessing/train_validation_test_distribution.png)

### Actual Project Findings:
- Chronological partitioning across 222 unique plant timestamp hours:
  - Train: Hours 0–154 (358 rows, 69.25%, March 11–17)
  - Validation: Hours 155–187 (92 rows, 17.79%, March 17–18)
  - Test: Hours 188–221 (67 rows, 12.96%, March 18–20)
- Exactly 0 overlapping timestamps between partitions.

### Viva Question & Answer:
**Examiner:** *"Why did you split by unique timestamp rather than slicing rows directly by row index?"*  
**Defense Answer:** *"Because multiple chillers operate concurrently and log at the exact same plant timestamp. If we sliced rows arbitrarily by index, records for Chiller 1 at 14:00 might end up in the training set while records for Chiller 2 at 14:00 end up in the test set. Because both chillers share the same ambient cooling water loop, this would cause cross-machine contemporaneous data leakage. Partitioning strictly on unique plant timestamps ensures that all concurrent plant operations at a given hour remain in the same partition."*

---

## Topic D: StandardScaler and Data Leakage Prevention

### Conceptual Foundation:
Z-score standardization transforms features to have zero mean and unit variance ($z = (x - \mu) / \sigma$). If the scaler is fitted on the entire dataset, the mean and standard deviation incorporate future distribution information, violating the causality requirement of prospective monitoring.

### Relevant Project File:
- [`data/processed/scaler.joblib`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/scaler.joblib)
- [`results/metrics/preprocessing/scaling_parameters.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/preprocessing/scaling_parameters.csv)

### Actual Project Findings:
- Scaler was fitted strictly on $X_{\text{train}}$ ($N = 358$).
- Standardized training feature means are $< 4.63 \times 10^{-15} \approx 0.0$, standard deviations are $1.000000$.
- Validation and test sets were transformed using the frozen training parameters.

### Viva Question & Answer:
**Examiner:** *"Why must we standardize the features before applying PCA in chiller monitoring?"*  
**Defense Answer:** *"Industrial chillers measure variables with wildly differing physical scales. Temperatures typically range from $5^\circ\text{C}$ to $90^\circ\text{C}$, whereas pressures range from $3\text{ kg/cm}^2$ to $15\text{ kg/cm}^2$. If PCA were applied to unscaled covariance, variables with large absolute numerical variance (like temperatures) would dominate the principal components, regardless of their physical significance. Standardizing to unit variance ensures that each sensor contributes proportionally based on its relative statistical variation."*

---

## Topic E & F: PCA Architecture, Eigenvalues, and Variance

### Conceptual Foundation:
PCA computes the eigendecomposition of the sample correlation matrix $\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$. The eigenvalues $\lambda_i$ represent the variance explained by each principal axis, and the eigenvectors $\mathbf{v}_i$ form an orthonormal basis ($\mathbf{V} \mathbf{V}^T = \mathbf{I}$).

### Relevant Project File:
- [`data/processed/pca_configuration.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/pca_configuration.json)
- [`results/metrics/pca_explained_variance.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/pca_explained_variance.csv)

### Actual Project Findings:
- Total features $p = 28$. Retained components $k = 12$.
- Cumulative variance explained: **85.46%**. Residual variance: **14.54%** across 16 residual dimensions.
- PC1 accounts for 23.77% ($\lambda_1 = 6.6755$); PC2 accounts for 13.64% ($\lambda_2 = 3.8307$).

### Viva Question & Answer:
**Examiner:** *"Why didn't you retain 25 components to capture 99% of the variance?"*  
**Defense Answer:** *"In reconstruction-error-based anomaly detection, retaining too many components defeats the detector. If we retain 99% of the variance, the model would reconstruct not only normal operational states but also anomalous deviations and sensor drift, causing the reconstruction error to remain small during a failure. By retaining $k = 12$ components (85.46% variance), we capture the true thermodynamic operating manifold while preserving 16 residual dimensions. When an anomalous event breaks physical correlations, the observation deviates into this 16-dimensional residual subspace, producing a sharp rise in reconstruction error."*

---

## Topic G & H: Reconstruction Error (SPE) and Threshold Calibration

### Conceptual Foundation:
The projection into the principal subspace is $\mathbf{z} = \mathbf{x} \mathbf{V}_k$, and the reconstruction is $\hat{\mathbf{x}} = \mathbf{z} \mathbf{V}_k^T$. The residual vector is $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$. The Squared Prediction Error (SPE or Q-statistic) is defined as:
$$\text{SPE} = \|\mathbf{e}\|_2^2 = \sum_{j=1}^{p} (x_j - \hat{x}_j)^2$$

### Relevant Project File:
- [`results/metrics/phase5/threshold_configuration.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/phase5/threshold_configuration.json)
- [`results/metrics/phase5/validation_threshold_comparison.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/phase5/validation_threshold_comparison.csv)

### Actual Project Findings:
- Validation SPE ($N_{\text{val}} = 92$): Median $= 2.2505$, 95th percentile $= \mathbf{9.2931}$.
- 99th percentile was distorted to $\tau = 1378.35$ due to a human manual logging typo ($322^\circ\text{C}$ in row 73).
- Selected primary threshold: $\tau = 9.2931$ (Empirical 95th percentile).

### Viva Question & Answer:
**Examiner:** *"Why did you use the 95th percentile instead of the 99th percentile for your anomaly threshold?"*  
**Defense Answer:** *"With a validation sample of $N_{\text{val}} = 92$, a 99th percentile represents a tail rank determined by less than one observation ($92 \times 0.01 = 0.92$). In our validation dataset, a human operator logged a discharge saturation temperature as $322^\circ\text{C}$ instead of $32.2^\circ\text{C}$, producing an extreme single-point SPE of $14,983$. This single outlier pulled the 99th percentile up to $1378.35$, which would completely desensitize the detector. In contrast, the 95th percentile ($\tau = 9.2931$) is anchored by lower order statistics (ranks 87 and 88), effectively isolating the true operational boundary while remaining robust against isolated recording errors."*

---

## Topic I, J & K: Synthetic Experiments, Metrics, and Confusion Matrices

### Conceptual Foundation:
When genuine ground-truth fault labels are unavailable in operational logs, controlled synthetic perturbations allow rigorous evaluation of detector sensitivity, precision, recall, and detection latency.

### Relevant Project File:
- [`results/metrics/final/final_experiment_comparison.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/final/final_experiment_comparison.csv)
- [`results/figures/final/final_confusion_matrices.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/final/final_confusion_matrices.png)

### Actual Project Findings:
- Scenario A (Spikes, $\pm 4.0\sigma$): Precision = 0.60, Recall = 1.00, F1 = 0.75, Event Recall = 100%, Delay = 0 hrs.
- Scenario B (Drift, $+0.7\sigma \to +3.5\sigma$): Precision = 0.33, Recall = 0.20, F1 = 0.25, Event Recall = 100%, Delay = 4 hrs.
- Scenario C (Relational Breaking, $-3.5\sigma$): Precision = 0.60, Recall = 1.00, F1 = 0.75, Event Recall = 100%, Delay = 0 hrs.

### Viva Question & Answer:
**Examiner:** *"Why was your observation-level recall only 20% in the gradual drift experiment, and why did you have 2 false positives in every scenario?"*  
**Defense Answer:** *"In Scenario B, drift was introduced as a linear ramp over 5 hours from $+0.7\sigma$ to $+3.5\sigma$. During the first 4 hours ($+0.7\sigma$ to $+2.8\sigma$), the perturbation remained within the normal statistical operating band ($\text{SPE} \le 9.2931$). It breached the threshold at Hour 5 when offset reached $+3.5\sigma$ ($\text{SPE} = 13.50$). Thus, early incipient points count as False Negatives at the row level, yielding 20% observation recall, but at the event level, detection was 100% with a 4-hour advance warning. Regarding False Positives: exactly two observations naturally exceeded the threshold in the clean test baseline (Index 38 on Chiller-4 and Index 54 on Chiller-5). Because synthetic ground truth labels all unmodified points as 0, these pre-existing baseline excursions appear as $FP = 2$. Net synthetic false positives were exactly ZERO across all three scenarios."*

---

## Topic L: Limitations and Future Improvements

### Conceptual Foundation:
Scientific integrity requires explicit recognition of model assumptions, sample constraints, and generalizability boundaries.

### Viva Question & Answer:
**Examiner:** *"What are the primary limitations of your study, and how would you improve it in future work?"*  
**Defense Answer:** *"The primary limitations are: (1) Sample size: our held-out test partition contains 67 observations across 34 plant hours, which demonstrates statistical validity but does not capture seasonal transitions across months. (2) Linear assumption: linear PCA assumes static linear correlations, whereas extreme cooling load swings involve non-linear heat transfer. (3) Absence of real fault logs: real maintenance work orders are unavailable, meaning clean test excursions are statistical anomalies rather than confirmed mechanical breakdowns. For future work, we recommend implementing Dynamic PCA (DPCA) to incorporate time-lagged variables, and exploring multi-model architectures that train chiller-specific baseline thresholds."*
