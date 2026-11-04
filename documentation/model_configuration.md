# Final PCA Model Architecture and Configuration Record

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Serialized Model Path:** `data/processed/pca_model.joblib`  
**Configuration File:** `data/processed/pca_configuration.json`  

---

## 1. Mathematical Architecture & Hyperparameters

- **Algorithm:** Linear Principal Component Analysis via Full Singular Value Decomposition (`scikit-learn PCA(svd_solver='full')`).
- **Whiten:** `False` (Component variance scaling preserved to maintain original residual energy geometry).
- **Random State:** Deterministic (Exact SVD on standardized sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{28 \times 28}$).
- **Training Observation Count ($N_{\text{train}}$):** 358 observations (Hours 0 to 154).
- **Input Dimension ($p$):** 28 continuous thermodynamic variables.
- **Retained Components ($k$):** **$12$ principal components**.
- **Residual Dimensionality ($p - k$):** **$16$ residual dimensions**.

---

## 2. Eigenvalue Spectrum & Variance Decomposition

The sample covariance matrix was decomposed into 28 orthogonal eigenvectors and eigenvalues ($\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$):

| Component | Eigenvalue ($\lambda_i$) | Singular Value ($s_i$) | Explained Variance Ratio | Cumulative Variance (%) | Kaiser Criterion ($\lambda_i \ge 1.0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **PC1** | 6.675534 | 48.817677 | 0.237746 | **23.77%** | PASS |
| **PC2** | 3.830682 | 36.980449 | 0.136428 | **37.42%** | PASS |
| **PC3** | 2.492886 | 29.832202 | 0.088783 | **46.30%** | PASS |
| **PC4** | 1.940246 | 26.318580 | 0.069101 | **53.21%** | PASS |
| **PC5** | 1.729975 | 24.851582 | 0.061623 | **59.37%** | PASS |
| **PC6** | 1.460596 | 22.836069 | 0.052028 | **64.57%** | PASS |
| **PC7** | 1.140231 | 20.177028 | 0.040618 | **68.63%** | PASS |
| **PC8** | 1.042838 | 19.297746 | 0.037146 | **72.35%** | PASS |
| **PC9** | 0.985619 | 18.761376 | 0.035105 | **75.86%** | Below 1.0 |
| **PC10** | 0.947230 | 18.392398 | 0.033739 | **79.23%** | Below 1.0 |
| **PC11** | 0.898846 | 17.915837 | 0.032015 | **82.40%** | Below 1.0 |
| **PC12** | 0.859598 | 17.520864 | 0.030619 | **85.46%** | Below 1.0 |
| **Residual (PC13–PC28)**| $\sum = 4.0945$ | — | 0.145390 | **14.54%** | Residual Space |

### Mathematical Rigor Checks:
- **Sum of Eigenvalues:** $\sum_{i=1}^{28} \lambda_i = 28.000000$ (Matches trace of standardized correlation matrix).
- **Component Orthogonality:** $\max \|\mathbf{V}\mathbf{V}^T - \mathbf{I}_{12 \times 12}\| = 1.998 \times 10^{-15}$ (Double-precision machine epsilon).
- **Invertibility Verification:** Training residual Mean Squared Error $= 0.145390$, exactly matching unexplained variance ratio ($1.0 - 0.85461 = 0.14539$).

---

## 3. Engineering Justification for Retaining $k = 12$ Components

1. **Industrial Variance Benchmark:** Retaining 12 components captures **85.46% cumulative variance**, satisfying the recommended $\ge 80\%$ standard for industrial chemical and mechanical monitoring.
2. **Kaiser-Guttman Rule Comparison:** The Kaiser rule ($\lambda \ge 1.0$) retains 8 components (72.35% variance). In our chiller dataset, retaining only 8 components leaves 27.65% unexplained variance, which is too coarse for normal steady-state operation and produces high false-alarm noise.
3. **Reconstruction Residual Reserve:** Retaining 12 components leaves **16 residual dimensions** (14.54% variance). In reconstruction-error anomaly detectors, if too many components are retained (e.g., $k = 25$, 99% variance), anomalies are reconstructed directly by the principal subspace, blinding the detector to failures. Retaining 12 components strikes the ideal balance between modeling normal operational variance and retaining residual sensitivity.

---

## 4. Physical Interpretation of Principal Subspace

- **PC1 (23.77% Variance) — High-Side Condenser Compression Lift:**
  - Positive loadings on `Discharge Saturation Temperature Comp 1` (+0.320) and `Comp 2` (+0.316).
  - Positive loadings on `Discharge Pressure Comp 1` (+0.301) and `Comp 2` (+0.303).
  - Positive loadings on `Chiller Lift Temperature Comp 1` (+0.285) and `Comp 2` (+0.284).
  - *Thermodynamic Meaning:* Captures overall plant compression ratio and heat rejection demand.
- **PC2 (13.64% Variance) — Low-Side Evaporator Cooling Load:**
  - Positive loadings on `Evaporator Water In Temperature` (+0.384) and `Evaporator Water Out Temperature` (+0.385).
  - Positive loading on `Suction Pressure` (+0.244).
  - Negative loading on `Suction Superheat` (-0.231).
  - *Thermodynamic Meaning:* Captures building cooling load and boiling heat transfer dynamics in the evaporator.

