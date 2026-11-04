# Phase 4 Working Document: PCA Model Development

**Document Purpose:** Factual engineering notes recording the implementation, mathematics, component selection analysis, training, and loading interpretation for the PCA-based multivariate detector.

---

## 1. Mathematical Foundations & Concept Reference

### A. Dimensionality Reduction
In high-dimensional multivariate monitoring, observing 28 correlated sensors independently fails to account for inter-sensor constraints. Dimensionality reduction maps the $p = 28$ correlated features into an orthogonal, lower-dimensional subspace of $k < p$ uncorrelated variables (principal components) that preserve the maximal variance of normal operation.

### B. Sample Covariance Matrix
Since the training observations $X_{\text{train}} \in \mathbb{R}^{358 \times 28}$ have been standardized to zero mean and unit variance ($\mu = 0, \sigma = 1$), the empirical sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{28 \times 28}$ equals the correlation matrix:
$$\mathbf{\Sigma} = \frac{1}{N_{\text{train}} - 1} X_{\text{train}}^T X_{\text{train}}$$

### C. Eigenvalue Decomposition
Principal Component Analysis solves the characteristic eigenvalue problem:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
where:
* $\lambda_i \ge 0$ is the $i$-th eigenvalue, representing the empirical variance captured along direction $\mathbf{v}_i$.
* $\mathbf{v}_i \in \mathbb{R}^{28}$ is the $i$-th orthonormal eigenvector ($\|\mathbf{v}_i\| = 1$, $\mathbf{v}_i^T \mathbf{v}_j = 0$ for $i \ne j$), defining the direction of the $i$-th principal component.

### D. Explained Variance Ratios
* Individual Explained Variance Ratio:
  $$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j} = \frac{\lambda_i}{28}$$
* Cumulative Explained Variance Ratio:
  $$\text{CEV}_k = \sum_{i=1}^{k} \text{EVR}_i$$

### E. Projection & Inverse Reconstruction
For any standardized observation vector $\mathbf{x} \in \mathbb{R}^{28}$:
* **Projection into Principal Subspace (Scores):**
  $$\mathbf{z} = \mathbf{x} \mathbf{V}_k \quad (\mathbf{z} \in \mathbb{R}^k)$$
  where $\mathbf{V}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{28 \times k}$ contains the first $k$ eigenvectors.
* **Inverse Reconstruction back to Original Space:**
  $$\hat{\mathbf{x}} = \mathbf{z} \mathbf{V}_k^T = \mathbf{x} \mathbf{V}_k \mathbf{V}_k^T \quad (\hat{\mathbf{x}} \in \mathbb{R}^{28})$$
* **Residual Vector (Anomaly Basis for Phase 5):**
  $$\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}} = \mathbf{x}(\mathbf{I} - \mathbf{V}_k \mathbf{V}_k^T)$$
  The squared Euclidean length $\|\mathbf{e}\|^2 = \sum_{j=1}^{28} (x_j - \hat{x}_j)^2$ is the **Squared Prediction Error (SPE)** / Reconstruction Error.

---

## 2. Training Data Verification

* Source: [`data/processed/train_features.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/train_features.csv)
* Observations ($N_{\text{train}}$): **358 observations** (strict training partition from March 11 to March 17, 10:00).
* Dimensionality ($p$): **28 core thermodynamic variables** (21 temperatures, 7 refrigerant pressures).
* Data Quality: 0 NaNs, 0 Infs, standardized ($\mu \approx 0, \sigma \approx 1$).
* Leakage Audit: Zero validation or test observations were seen during PCA training.

---

## 3. Full Spectrum Explained Variance (Training Partition)

All 28 eigenvalues computed by full SVD:

| Component | Eigenvalue ($\lambda_i$) | Individual EV (%) | Cumulative EV (%) | Kaiser Criterion ($\lambda \ge 1.0$) |
| :--- | :--- | :--- | :--- | :--- |
| **PC 1** | **6.6755** | **23.77%** | **23.77%** | Pass |
| **PC 2** | **3.8307** | **13.64%** | **37.42%** | Pass |
| **PC 3** | **2.4929** | **8.88%** | **46.30%** | Pass |
| **PC 4** | **1.9402** | **6.91%** | **53.21%** | Pass |
| **PC 5** | **1.7300** | **6.16%** | **59.37%** | Pass |
| **PC 6** | **1.4606** | **5.20%** | **64.57%** | Pass |
| **PC 7** | **1.1402** | **4.06%** | **68.63%** | Pass |
| **PC 8** | **1.0401** | **3.70%** | **72.33%** | Pass |
| **PC 9** | 0.9832 | 3.50% | 75.84% | Fail |
| **PC 10** | 0.9397 | 3.35% | 79.18% | Fail |
| **PC 11** | 0.9318 | 3.32% | 82.50% | Fail (~80% Threshold) |
| **PC 12** | **0.8312** | **2.96%** | **85.46%** | **Fail (~85% Recommended)** |
| **PC 13** | 0.7838 | 2.79% | 88.25% | Fail |
| **PC 14** | 0.7298 | 2.60% | 90.85% | Fail (~90% Threshold) |
| **PC 15** | 0.5538 | 1.97% | 92.82% | Fail |
| **PC 16** | 0.4342 | 1.55% | 94.37% | Fail |
| **PC 17** | 0.3641 | 1.30% | 95.67% | Fail (~95% Threshold) |
| **PC 18** | 0.3137 | 1.12% | 96.78% | Fail |
| **PC 19–28** | $< 0.20$ | $< 0.75\%$ each | 97.51% $\rightarrow$ 100.0% | Fail |

*Artifact Saved:* [`results/metrics/pca_explained_variance.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/pca_explained_variance.csv)

---

## 4. Component Selection & Trade-Off Analysis

| Strategy | Retained Components ($k$) | Cumulative Variance (%) | Residual Dimensions ($28 - k$) | Residual Variance (%) | Technical Trade-off for Anomaly Detection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kaiser Rule ($\lambda \ge 1.0$)** | 8 | 72.33% | 20 | 27.67% | Discards >27% of normal physical variance, causing high baseline SPE noise on normal validation data. |
| **80% Target** | 11 | 82.50% | 17 | 17.50% | Acceptable compression, but slightly under-fits coupled secondary circuits. |
| **85% Target (RECOMMENDED)** | **12** | **85.46%** | **16** | **14.54%** | **Optimal MSPC Balance:** Captures >85% of normal thermodynamic covariance while reserving 16 residual dimensions to catch anomalous decoupling. |
| **90% Target** | 14 | 90.85% | 14 | 9.15% | High fidelity; suitable if measurement noise is extremely low. |
| **95% Target** | 17 | 95.67% | 11 | 4.33% | Over-retention: Higher components begin learning noise, reducing anomaly contrast in residual SPE. |

*Artifact Saved:* [`results/metrics/pca_component_selection_comparison.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/pca_component_selection_comparison.csv)

---

## 5. Final Model Configuration

* **Model Type:** `sklearn.decomposition.PCA`
* **Solver:** `svd_solver='full'`, `whiten=False`
* **Input Features ($p$):** 28
* **Retained Components ($k$):** **12**
* **Cumulative Explained Variance:** **85.46%**
* **Residual Degrees of Freedom:** **16 dimensions (14.54% variance)**
* **Model Artifact:** [`data/processed/pca_model.joblib`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/pca_model.joblib)
* **Metadata Configuration:** [`data/processed/pca_configuration.json`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/data/processed/pca_configuration.json)

---

## 6. Component Loading Interpretation (The Physical Subspaces)

### Principal Component 1: High-Side / Condenser Compression Work Subspace (23.77% Variance)
* **Dominant Features:**
  - `Discharge Saturation Temperature Comp 1`: Loading = $+0.3488$
  - `Chiller Lift Pressure Comp 1`: Loading = $+0.3442$
  - `Chiler Lift Temperature Comp 1`: Loading = $+0.3008$
  - `Liquid Temperature Comp 2`: Loading = $+0.3002$
  - `EXV Differential Pressure Comp 1`: Loading = $+0.2768$
* **Physical Interpretation:** PC1 captures the common high-pressure lift work of the compression cycle. When condensing temperature and lift pressure rise together, PC1 increases. Both temperature and pressure load strongly in the same direction, empirically proving their multivariate coupling.

### Principal Component 2: Low-Side / Evaporator Cooling Load Subspace (13.64% Variance)
* **Dominant Features:**
  - `Suction Pressure Kg /cm 2`: Loading = $+0.4562$
  - `Suction Saturation Temperature Deg C`: Loading = $+0.3769$
  - `Suction Temperature Deg C`: Loading = $+0.3394$
  - `Discharge Superheat Comp 2`: Loading = $-0.3321$
  - `Condenser Approach Comp 2`: Loading = $+0.3102$
* **Physical Interpretation:** PC2 captures the evaporator operating condition and building thermal cooling demand. Suction pressure and suction temperatures load positively together, while superheat loads negatively (higher evaporating load draws denser vapor, lowering superheat).

*Artifacts Saved:*
* Loadings Matrix: [`results/metrics/pca_component_loadings.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/pca_component_loadings.csv)
* Training Scores: [`results/metrics/pca_training_scores.csv`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/metrics/pca_training_scores.csv)

---

## 7. Diagnostic Visualizations Generated (All 300 DPI)

1. **Scree Plot:** [`results/figures/pca/scree_plot.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/scree_plot.png) — Displays eigenvalues per component with the Kaiser threshold ($\lambda = 1.0$) and elbow transition.
2. **Cumulative Explained Variance Curve:** [`results/figures/pca/cumulative_explained_variance.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/cumulative_explained_variance.png) — Visualizes the cumulative variance curve with candidate cutoffs (85%, 90%).
3. **Component Loading Heatmap:** [`results/figures/pca/component_loadings.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/component_loadings.png) — Complete $28 \times 12$ matrix of eigenvector weights.
4. **Component Contribution Bar Chart:** [`results/figures/pca/component_contribution.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/component_contribution.png) — Individual variance percentage per retained component.
5. **Training Component Scores Scatter (PC1 vs PC2):** [`results/figures/pca/training_component_scores.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/training_component_scores.png) — Scatter of training observations in principal subspace.
6. **Chiller-Specific Component Diagnostics:** [`results/figures/pca/chiller_component_diagnostics.png`](file:///c:/Users/agrbr/OneDrive/Desktop/Chiller-Anomaly-Detection-Official/results/figures/pca/chiller_component_diagnostics.png) — Confirms that all 5 chillers form a unified, continuous operating cluster in PC1–PC2 space without isolated machine clusters.

