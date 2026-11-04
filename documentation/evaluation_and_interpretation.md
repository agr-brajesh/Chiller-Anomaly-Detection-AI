# Final Model Evaluation, Detection Performance, and Interpretation Record

**Project Title:** AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data  
**Student Register Number:** 24BCE0551  
**Track:** Multivariate and Time-Series Detection  
**Assigned Method:** PCA-Based Multivariate Anomaly Detector  
**Validation Threshold:** $\tau = 9.2931$ (Empirical 95th Percentile of Validation SPE)  
**Test Partition:** Held-out operational test set ($N_{\text{test}} = 67$, March 18 to 20)  

---

## 1. Clean Operational Test Evaluation

- **Test Observations:** $N_{\text{test}} = 67$ across 34 unique plant hours.
- **Normal Operating Observations ($\text{SPE} \le \tau$):** 65 out of 67 (**97.01%**).
- **Flagged Potential Anomalies ($\text{SPE} > \tau$):** Exactly 2 observations (**2.99%**).
- **Clean Test Error Statistics:**
  - Mean SPE: $2.4913$
  - Median SPE: $1.9749$
  - Minimum SPE: $0.6425$ (Chiller-3)
  - Maximum SPE: $13.5997$ (Chiller-5)

### Analysis of Flagged Operational Excursions:
1. **Chiller - 4 at 2026-03-19 17:00:00 (Index 38):**
   - $\text{SPE} = 9.9126$ (Exceeds threshold $9.2931$).
   - Top contributor: `Suction Temperature Deg C` (46.90% of SPE, $z = +2.65$).
   - Physical context: Suction line ran warmer than expected under cold chilled water leaving conditions, indicating abnormal vapor superheating.
2. **Chiller - 5 at 2026-03-20 02:00:00 (Index 54):**
   - $\text{SPE} = 13.5997$ (Highest error in test set).
   - Top contributor: `Chiler Lift Temperature Comp 1` (27.79% of SPE, $z = -4.56$).
   - Physical context: Compressor 1 experienced an acute drop in temperature lift while evaporator chilled water outlet was elevated ($z = +2.17$), indicating compressor unloading or valve bypass.

---

## 2. Controlled Synthetic Anomaly Experiment Performance

To evaluate the detector against defined ground truth, three independent perturbation scenarios were tested on the test partition:

| Metric Category | Scenario A: Sudden Spikes | Scenario B: Gradual Drift | Scenario C: Relationship Breaking |
| :--- | :---: | :---: | :---: |
| **Physical Fault Model** | Transducer / wiring spike ($\pm 4.0\sigma$) | Calibration drift ($+0.7\sigma \to +3.5\sigma$) | Decoupling discharge temp & press ($-3.5\sigma$) |
| **Duration** | 1 hour per event (3 events) | 5 continuous hours (1 event) | 3 continuous hours (1 event) |
| **Total Test Observations** | 67 | 67 | 67 |
| **Injected Observations** | 3 | 5 | 3 |
| **True Positives (TP)** | 3 | 1 | 3 |
| **False Negatives (FN)** | 0 | 4 | 0 |
| **False Positives (FP)** | 2 (Baseline clean flags) | 2 (Baseline clean flags) | 2 (Baseline clean flags) |
| **True Negatives (TN)** | 62 | 60 | 62 |
| **Precision** | **0.6000** | **0.3333** | **0.6000** |
| **Recall** | **1.0000** | **0.2000** | **1.0000** |
| **F1-Score** | **0.7500** | **0.2500** | **0.7500** |
| **Accuracy** | **0.9701** | **0.9104** | **0.9701** |
| **False Positive Rate (FPR)** | 0.0312 | 0.0323 | 0.0312 |
| **Total Events Injected** | 3 | 1 | 1 |
| **Events Detected** | 3 | 1 | 1 |
| **Event Detection Rate (%)** | **100.0%** | **100.0%** | **100.0%** |
| **Detection Delay** | **0 hours (Immediate)** | **4 hours (At $+3.5\sigma$)** | **0 hours (Immediate)** |
| **Net Synthetic False Positives**| **0** | **0** | **0** |

---

## 3. Key Statistical and Operational Insights

1. **Why False Positives Equal Exactly 2 Across All Scenarios:**
   The two false positives in each scenario represent the exact two naturally occurring operational excursions present in the clean test baseline (Indices 38 and 54). Because synthetic ground truth marks unmodified observations as label $0$, these pre-existing baseline excursions are counted as False Positives. **Net synthetic false positives across all scenarios: ZERO.**
2. **Observation-Level vs Event-Level Trade-Off in Gradual Drift:**
   In Scenario B, observation-level recall is $20.0\%$ ($1/5$) because the initial 4 hours of incipient drift ($+0.7\sigma, +1.4\sigma, +2.1\sigma, +2.8\sigma$) remained within the normal operational tolerance band ($\text{SPE} \le 9.2931$). However, at the event level, the developing fault was **100% detected** at Hour 5 ($+3.5\sigma$, $\text{SPE} = 13.50$), providing operators with actionable advance notice before catastrophic failure.
3. **Conclusive Answer to the Research Question:**
   Scenario C directly decoupled the strongly correlated physical pair `Discharge Temperature` and `Discharge Pressure` ($r = +0.9578$). The model responded with immediate, $100\%$ detection across all 3 affected observations with zero latency, providing direct mathematical and empirical proof that anomalies can be detected from correlated temperature and pressure variables.

---

## 4. Scope and Limitations Summary

- **Absence of Real Fault Labels:** Real maintenance logs do not exist in the official operational dataset. Clean test flags represent statistical anomalies, not confirmed mechanical failures.
- **Linear Model Boundary:** Linear PCA models static covariance. While local R134a vapor-pressure relationships are highly collinear, extreme operational transitions across seasons may require non-linear extensions.
- **Small Test Window:** $N_{\text{test}} = 67$ across 34 hours provides rigorous proof of concept but limits multi-month seasonal benchmarking.

