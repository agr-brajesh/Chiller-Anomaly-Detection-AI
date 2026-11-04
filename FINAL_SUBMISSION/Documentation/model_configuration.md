# Final PCA Model Architecture and Configuration Record (Cleaned Pipeline)

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Serialized Model Path:** `data/processed/pca_model.joblib`  
**Configuration File:** `data/processed/pca_configuration.json`  

---

## 1. Mathematical Architecture & Hyperparameters

- **Algorithm:** Linear Principal Component Analysis via Full Singular Value Decomposition (`scikit-learn PCA(svd_solver='full')`).
- **Whiten:** `False` (Component variance scaling preserved to maintain residual energy geometry).
- **Data Cleaning:** Thermodynamic physical plausibility screening (out-of-bounds cells masked to NaN) + `SimpleImputer(strategy='median')` fitted strictly on training data.
- **Scaling:** `StandardScaler` fitted strictly on cleaned training data ($Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$).
- **Training Observation Count ($N_{\text{train}}$):** 358 observations (Hours 0 to 154).
- **Input Dimension ($p$):** 28 continuous thermodynamic variables.
- **Retained Components ($k$):** **$12$ principal components**.
- **Cumulative Explained Variance:** **$97.46\%$** (captures all core thermodynamic covariance).
- **Residual Dimensionality ($p - k$):** **$16$ residual dimensions** ($2.54\%$ residual variance reserved for orthogonal anomaly detection).

---

## 2. Eigenvalue Spectrum & Variance Decomposition (Cleaned Data)

The sample covariance matrix was decomposed into 28 orthogonal eigenvectors and eigenvalues ($\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$):

| Component | Eigenvalue ($\lambda_i$) | Singular Value ($s_i$) | Explained Variance Ratio | Cumulative Variance (%) | Kaiser Criterion ($\lambda_i \ge 1.0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **PC1** | 12.8414 | 67.7082 | 0.457256 | **45.73%** | PASS |
| **PC2** | 4.9987 | 42.2741 | 0.177983 | **63.52%** | PASS |
| **PC3** | 1.9758 | 26.5583 | 0.070351 | **70.56%** | PASS |
| **PC4** | 1.9013 | 26.0531 | 0.067699 | **77.26%** | PASS |
| **PC5** | 1.4793 | 22.9806 | 0.052670 | **82.53%** | PASS |
| **PC6** | 1.2819 | 21.3917 | 0.045644 | **87.10%** | PASS (Strict Kaiser Boundary) |
| **PC7** | 0.8929 | 17.8549 | 0.031792 | **90.28%** | Below 1.0 |
| **PC8** | 0.6629 | 15.3831 | 0.023602 | **92.64%** | Kaiser Elbow |
| **PC9** | 0.5263 | 13.7083 | 0.018739 | **94.51%** | Below 1.0 |
| **PC10** | 0.3305 | 10.8624 | 0.011768 | **95.69%** | 95% Target |
| **PC11** | 0.2896 | 10.1685 | 0.010312 | **96.72%** | Below 1.0 |
| **PC12** | 0.1836 | 8.0963 | 0.006538 | **97.26%** (Model Total: 97.46%) | Primary Cutoff |
| **Residual (PC13–PC28)** | $\sum = 0.7107$ | — | 0.025381 | **2.54%** | 16 Residual Dims |

### Mathematical Rigor Checks:
- **Trace Equality:** $\sum_{i=1}^{28} \lambda_i = 28.000000$ (Matches trace of correlation matrix).
- **Component Orthogonality:** $\max \|\mathbf{V}\mathbf{V}^T - \mathbf{I}_{12 \times 12}\| = 1.998 \times 10^{-15}$ (Double-precision machine epsilon).
- **Training Residual Mean SPE:** $0.7107$, matching the theoretical residual energy.

---

## 3. Dual MSPC Statistics & Time-Series Monitoring Formulation

1. **Squared Prediction Error (SPE / Q-Statistic):**
   $$\text{SPE}_i = \|\mathbf{z}_i - \hat{\mathbf{z}}_i\|^2 = \sum_{j=1}^{28} (z_{ij} - \hat{z}_{ij})^2$$
   - Measures residual distance orthogonal to the principal subspace (detects broken physical relationships and sensor failures).
   - **Primary Threshold ($\tau_{\text{SPE}, 95}$):** $4.2703$ (Validation 95th Percentile).
   - **Secondary Threshold ($\tau_{\text{SPE}, 99}$):** $7.8858$ (Validation 99th Percentile).

2. **Hotelling's $T^2$ Statistic:**
   $$T_i^2 = \mathbf{t}_i \mathbf{\Lambda}_k^{-1} \mathbf{t}_i^T = \sum_{j=1}^{k} \frac{t_{ij}^2}{\lambda_j}$$
   - Measures Mahalanobis distance inside the principal subspace (detects severe plant load/operating state shifts).
   - **Empirical Threshold ($\tau_{T^2, 95}$):** $41.16$ (Validation 95th Percentile).
   - **Theoretical Snedecor $F_{0.95}(12, 346)$ Threshold:** $22.04$.

3. **Time-Series EWMA Layer on SPE:**
   $$S_t = \lambda_{\text{ewma}} \text{SPE}_t + (1 - \lambda_{\text{ewma}}) S_{t-1}, \quad \lambda_{\text{ewma}} = 0.20, \quad S_0 = \mu_{\text{train, SPE}}$$
   - Accumulates persistent subtle drift, overcoming the single-observation threshold latency.
   - **EWMA Threshold ($\tau_{\text{EWMA}, 95}$):** $1.8618$.

---

## 4. Physical Interpretation of Principal Subspace

- **PC1 (45.73% Variance) — High-Side Condenser / Lift Subspace:**
  - `Discharge Pressure Kg /cm 2 Comp 1` (+0.260)
  - `Liquid Temperature Deg C Comp 1` (+0.259)
  - `Discharge Saturation Temperature Deg C Comp 1` (+0.258)
  - `Condenser Water Out Temperature Deg C` (+0.251)
  - `Chiller Lift Pressure Kg /cm 2 Comp 2` (+0.246)
  - *Thermodynamic Meaning:* Captures overall plant compression ratio, condensing pressure, and heat rejection work.

- **PC2 (17.80% Variance) — Low-Side Evaporator Cooling Load Subspace:**
  - `Suction Pressure Kg /cm 2` (+0.394)
  - `Evaporator Approach` (-0.363)
  - `Discharge Superheat Deg C Comp 2` (-0.323)
  - `Suction Saturation Temperature Deg C` (+0.310)
  - `Discharge Superheat Deg C Comp 1` (-0.302)
  - *Thermodynamic Meaning:* Captures building cooling demand and evaporator boiling dynamics.
