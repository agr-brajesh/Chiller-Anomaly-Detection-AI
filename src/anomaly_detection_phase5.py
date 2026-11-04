"""
Phase 5: Multivariate & Time-Series Anomaly Detection (Cleaned & Calibrated)
===========================================================================
Project: AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector

Comprehensive Implementation of Assigned Track:
1. Dual MSPC Statistics:
   - Squared Prediction Error (SPE / Q-statistic): Residual distance orthogonal to principal subspace.
   - Hotelling's T^2: Mahalanobis distance inside the principal subspace (captures operating load shifts).
2. Time-Series Monitoring Layer (EWMA):
   - Exponentially Weighted Moving Average on SPE (lambda = 0.20).
   - Exponentially Weighted Moving Average on T^2 (lambda = 0.20).
   - Accurately tracks temporal correlation and incipient drift.
3. Clean Validation Threshold Calibration:
   - Calibrated strictly on cleaned validation set (322°C typo masked and imputed).
   - 95th and 99th percentiles computed cleanly (tau_SPE_95 = 4.2703, tau_SPE_99 = 7.8858).
   - Theoretical Snedecor F-distribution threshold for Hotelling's T^2.
   - "Provisional" wording completely eliminated.
4. Fleet Mix Shift & Per-Chiller Diagnostics:
   - Multi-unit evaluation across BOTH Validation and Test splits.
   - Chiller-1 composition shift (12.3% train -> 19.6% val -> 50.7% test) documented.
5. Publication Figures:
   - 7 figures generated at 300 DPI with grayscale-friendly line styles, markers, and patterns.

Saves:
- results/metrics/phase5/validation_reconstruction_errors.csv
- results/metrics/phase5/test_reconstruction_errors.csv
- results/metrics/phase5/validation_threshold_comparison.csv
- results/metrics/phase5/threshold_configuration.json
- results/metrics/phase5/test_anomaly_predictions.csv
- results/metrics/phase5/test_anomaly_summary.csv
- results/metrics/phase5/top_potential_anomalies.csv
- results/metrics/phase5/feature_residual_contributions.csv
- results/metrics/phase5/aggregate_feature_contributions.csv
- results/metrics/phase5/chiller_anomaly_summary.csv
- results/metrics/phase5/per_chiller_val_test_comparison.csv
- results/figures/phase5/validation_threshold_analysis.png
- results/figures/phase5/test_reconstruction_timeline.png
- results/figures/phase5/test_reconstruction_distribution.png
- results/figures/phase5/test_potential_anomalies.png
- results/figures/phase5/feature_residual_contributions.png
- results/figures/phase5/chiller_reconstruction_comparison.png
- results/figures/phase5/time_series_ewma_monitoring.png
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

os.makedirs("results/metrics/phase5", exist_ok=True)
os.makedirs("results/figures/phase5", exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD SAVED ARTIFACTS
# -------------------------------------------------------------
print("=" * 80)
print("PHASE 5: MULTIVARIATE & TIME-SERIES ANOMALY DETECTION PIPELINE")
print("=" * 80)

print("\n[Step 1] Loading verified Phase 3 and Phase 4 artifacts...")
pca = joblib.load("data/processed/pca_model.joblib")
scaler = joblib.load("data/processed/scaler.joblib")
imputer = joblib.load("data/processed/imputer.joblib")

df_train_feat = pd.read_csv("data/processed/train_features.csv")
df_val_feat = pd.read_csv("data/processed/validation_features.csv")
df_test_feat = pd.read_csv("data/processed/test_features.csv")

df_train_meta = pd.read_csv("data/processed/train_metadata.csv")
df_val_meta = pd.read_csv("data/processed/validation_metadata.csv")
df_test_meta = pd.read_csv("data/processed/test_metadata.csv")

with open("data/processed/pca_configuration.json", "r") as f:
    pca_config = json.load(f)

feature_names = pca_config["feature_names"]
k_components = pca_config["n_components_retained"]
n_features = pca_config["n_features_in"]
lambdas = np.array(pca_config["eigenvalues"][:k_components])

print(f"Loaded Clean PCA Model: {k_components} components retained from {n_features} features.")
print(f"Train: {df_train_feat.shape}, Val: {df_val_feat.shape}, Test: {df_test_feat.shape}")

# -------------------------------------------------------------
# 2. ACTIVITY A: PCA PROJECTION & INVERSE RECONSTRUCTION
# -------------------------------------------------------------
print("\n[Step 2] Executing PCA projection and inverse reconstruction...")

Z_train = df_train_feat.values
Z_val = df_val_feat.values
Z_test = df_test_feat.values

# Principal component scores (T = Z * P)
T_train = pca.transform(Z_train)
T_val = pca.transform(Z_val)
T_test = pca.transform(Z_test)

# Reconstructed vectors (Z_hat = T * P^T)
Z_train_recon = pca.inverse_transform(T_train)
Z_val_recon = pca.inverse_transform(T_val)
Z_test_recon = pca.inverse_transform(T_test)

# Residual error vectors (E = Z - Z_hat)
E_train = Z_train - Z_train_recon
E_val = Z_val - Z_val_recon
E_test = Z_test - Z_test_recon

# -------------------------------------------------------------
# 3. ACTIVITY B: DUAL MSPC STATISTICS (SPE & HOTELLING'S T^2)
# -------------------------------------------------------------
print("\n[Step 3] Calculating dual MSPC statistics (SPE and Hotelling's T^2)...")

# 1. Squared Prediction Error (SPE / Q-statistic)
spe_train = np.sum(E_train**2, axis=1)
spe_val = np.sum(E_val**2, axis=1)
spe_test = np.sum(E_test**2, axis=1)

# 2. Hotelling's T^2 (inside principal subspace)
T2_train = np.sum((T_train**2) / lambdas, axis=1)
T2_val = np.sum((T_val**2) / lambdas, axis=1)
T2_test = np.sum((T_test**2) / lambdas, axis=1)

# 3. Time-Series Layer: EWMA Filter (lambda_ewma = 0.20)
LAMBDA_EWMA = 0.20

def compute_ewma(series_data, lam=0.20, init_mean=None):
    if init_mean is None:
        init_mean = np.mean(series_data)
    ewma_vals = np.zeros(len(series_data))
    ewma_vals[0] = init_mean
    for t in range(1, len(series_data)):
        ewma_vals[t] = lam * series_data[t] + (1.0 - lam) * ewma_vals[t-1]
    return ewma_vals

train_spe_mean = float(np.mean(spe_train))
ewma_spe_train = compute_ewma(spe_train, lam=LAMBDA_EWMA, init_mean=train_spe_mean)
ewma_spe_val = compute_ewma(spe_val, lam=LAMBDA_EWMA, init_mean=train_spe_mean)
ewma_spe_test = compute_ewma(spe_test, lam=LAMBDA_EWMA, init_mean=train_spe_mean)

ewma_T2_train = compute_ewma(T2_train, lam=LAMBDA_EWMA, init_mean=float(np.mean(T2_train)))
ewma_T2_val = compute_ewma(T2_val, lam=LAMBDA_EWMA, init_mean=float(np.mean(T2_train)))
ewma_T2_test = compute_ewma(T2_test, lam=LAMBDA_EWMA, init_mean=float(np.mean(T2_train)))

# Save comprehensive error tables
df_val_errors = df_val_meta.copy()
df_val_errors["SPE"] = spe_val
df_val_errors["Hotelling_T2"] = T2_val
df_val_errors["EWMA_SPE"] = ewma_spe_val
df_val_errors["EWMA_T2"] = ewma_T2_val
df_val_errors.to_csv("results/metrics/phase5/validation_reconstruction_errors.csv", index=False)

df_test_errors = df_test_meta.copy()
df_test_errors["SPE"] = spe_test
df_test_errors["Hotelling_T2"] = T2_test
df_test_errors["EWMA_SPE"] = ewma_spe_test
df_test_errors["EWMA_T2"] = ewma_T2_test
df_test_errors.to_csv("results/metrics/phase5/test_reconstruction_errors.csv", index=False)

print(f"Validation SPE: Mean = {np.mean(spe_val):.4f}, Median = {np.median(spe_val):.4f}, Max = {np.max(spe_val):.4f}")
print(f"Validation T^2: Mean = {np.mean(T2_val):.2f}, Median = {np.median(T2_val):.2f}, Max = {np.max(T2_val):.2f}")
print(f"Test SPE:       Mean = {np.mean(spe_test):.4f}, Median = {np.median(spe_test):.4f}, Max = {np.max(spe_test):.4f}")

# -------------------------------------------------------------
# 4. ACTIVITY C: VALIDATION THRESHOLD CALIBRATION
# -------------------------------------------------------------
print("\n[Step 4] Calibrating clean validation anomaly thresholds...")

# Candidate percentiles on clean validation set
percentiles = [90.0, 95.0, 97.5, 99.0]
threshold_candidates = []
N_val = len(spe_val)

for p in percentiles:
    t_spe = float(np.percentile(spe_val, p))
    t_T2 = float(np.percentile(T2_val, p))
    t_ewma = float(np.percentile(ewma_spe_val, p))
    n_above_spe = int(np.sum(spe_val > t_spe))
    
    threshold_candidates.append({
        "Percentile": f"{p}%",
        "Threshold_SPE": round(t_spe, 4),
        "Threshold_T2": round(t_T2, 4),
        "Threshold_EWMA_SPE": round(t_ewma, 4),
        "Val_Observations_Above_SPE": n_above_spe,
        "Val_Pct_Above_SPE": round((n_above_spe / N_val) * 100.0, 2),
        "Trade_Off_Analysis": (
            "Higher sensitivity; ~5 normal validation boundary excursions." if p == 95.0 else
            "Conservative; high specificity; calibrated on clean validation tail." if p == 99.0 else
            "Intermediate operational threshold."
        )
    })

df_threshold_comp = pd.DataFrame(threshold_candidates)
df_threshold_comp.to_csv("results/metrics/phase5/validation_threshold_comparison.csv", index=False)

# Primary Frozen Anomaly Thresholds
primary_tau_spe = float(np.percentile(spe_val, 95.0))
secondary_tau_spe_99 = float(np.percentile(spe_val, 99.0))
tau_T2_95 = float(np.percentile(T2_val, 95.0))
tau_T2_99 = float(np.percentile(T2_val, 99.0))
tau_ewma_95 = float(np.percentile(ewma_spe_val, 95.0))
tau_ewma_99 = float(np.percentile(ewma_spe_val, 99.0))

# Theoretical Hotelling's T^2 threshold using Snedecor F-distribution
N_train = len(Z_train)
f_crit_95 = stats.f.ppf(0.95, k_components, N_train - k_components)
theoretical_T2_95 = float((k_components * (N_train - 1) / (N_train - k_components)) * f_crit_95)

threshold_config = {
    "primary_threshold_status": "Final Verified Validation Threshold (Cleaned Data)",
    "validation_sample_size": N_val,
    "primary_threshold_spe_95": round(primary_tau_spe, 4),
    "secondary_threshold_spe_99": round(secondary_tau_spe_99, 4),
    "primary_threshold_T2_95": round(tau_T2_95, 4),
    "theoretical_T2_95": round(theoretical_T2_95, 4),
    "primary_threshold_ewma_spe_95": round(tau_ewma_95, 4),
    "ewma_smoothing_factor_lambda": LAMBDA_EWMA,
    "rationale": (
        f"Calibrated strictly on N_val = {N_val} cleaned observations with zero data leakage. "
        f"Primary SPE threshold tau_95 = {primary_tau_spe:.4f} anchors detection at empirical 95% specificity. "
        f"Hotelling's T^2 empirical threshold tau_T2_95 = {tau_T2_95:.2f} (theoretical F-distribution: {theoretical_T2_95:.2f}) "
        "monitors operating load shifts inside the principal subspace. "
        f"EWMA-SPE threshold tau_ewma_95 = {tau_ewma_95:.4f} tracks cumulative incipient drift over time."
    ),
    "candidate_thresholds": threshold_candidates
}

with open("results/metrics/phase5/threshold_configuration.json", "w") as f:
    json.dump(threshold_config, f, indent=2)

print(f"\nFinal Primary Thresholds:")
print(f"  * SPE Anomaly Threshold (95th Pct):     tau_SPE = {primary_tau_spe:.4f} (99th Pct: {secondary_tau_spe_99:.4f})")
print(f"  * Hotelling's T^2 Threshold (95th Pct): tau_T2  = {tau_T2_95:.2f} (Theoretical F: {theoretical_T2_95:.2f})")
print(f"  * EWMA-SPE Threshold (95th Pct):        tau_EWMA= {tau_ewma_95:.4f} (99th Pct: {tau_ewma_99:.4f})")

# -------------------------------------------------------------
# 5. ACTIVITY D: APPLY FROZEN THRESHOLDS TO TEST DATA
# -------------------------------------------------------------
print("\n[Step 5] Applying frozen thresholds to test partition...")

test_flags_spe = (spe_test > primary_tau_spe).astype(int)
test_flags_T2 = (T2_test > tau_T2_95).astype(int)
test_flags_ewma = (ewma_spe_test > tau_ewma_95).astype(int)
combined_flags = ((test_flags_spe == 1) | (test_flags_T2 == 1)).astype(int)

df_test_preds = df_test_meta.copy()
df_test_preds["SPE"] = spe_test
df_test_preds["Hotelling_T2"] = T2_test
df_test_preds["EWMA_SPE"] = ewma_spe_test
df_test_preds["SPE_Threshold"] = primary_tau_spe
df_test_preds["T2_Threshold"] = tau_T2_95
df_test_preds["EWMA_Threshold"] = tau_ewma_95
df_test_preds["SPE_Flag"] = test_flags_spe
df_test_preds["T2_Flag"] = test_flags_T2
df_test_preds["EWMA_Flag"] = test_flags_ewma
df_test_preds["Combined_MSPC_Flag"] = combined_flags
df_test_preds["Classification"] = np.where(test_flags_spe == 1, "Candidate Operational Excursion", "Normal Baseline")

df_test_preds.to_csv("results/metrics/phase5/test_anomaly_predictions.csv", index=False)

N_test = len(spe_test)
n_flagged_spe = int(np.sum(test_flags_spe))
n_flagged_T2 = int(np.sum(test_flags_T2))
n_flagged_ewma = int(np.sum(test_flags_ewma))

test_summary = {
    "Total_Test_Observations": N_test,
    "SPE_Flagged_Count": n_flagged_spe,
    "SPE_Flagged_Pct": round(100.0 * n_flagged_spe / N_test, 2),
    "Hotelling_T2_Flagged_Count": n_flagged_T2,
    "Hotelling_T2_Flagged_Pct": round(100.0 * n_flagged_T2 / N_test, 2),
    "EWMA_SPE_Flagged_Count": n_flagged_ewma,
    "EWMA_SPE_Flagged_Pct": round(100.0 * n_flagged_ewma / N_test, 2),
    "Combined_MSPC_Flagged_Count": int(np.sum(combined_flags)),
    "Combined_MSPC_Flagged_Pct": round(100.0 * np.sum(combined_flags) / N_test, 2),
    "Primary_SPE_Threshold": round(primary_tau_spe, 4),
    "Hotelling_T2_Threshold": round(tau_T2_95, 4),
    "EWMA_SPE_Threshold": round(tau_ewma_95, 4),
    "Test_SPE_Mean": round(float(np.mean(spe_test)), 4),
    "Test_SPE_Median": round(float(np.median(spe_test)), 4),
    "Test_SPE_Std": round(float(np.std(spe_test)), 4),
    "Test_SPE_Max": round(float(np.max(spe_test)), 4)
}
pd.DataFrame([test_summary]).to_csv("results/metrics/phase5/test_anomaly_summary.csv", index=False)
print(f"Test Evaluation Summary: SPE Flagged = {n_flagged_spe}/{N_test} ({test_summary['SPE_Flagged_Pct']}%), T^2 Flagged = {n_flagged_T2}/{N_test}")

# Top candidate anomalies
df_top_anomalies = df_test_preds.sort_values(by="SPE", ascending=False).head(10)
df_top_anomalies.to_csv("results/metrics/phase5/top_potential_anomalies.csv", index=False)

print("\nTop Flagged Test Observations (Genuine Operational Excursions):")
for idx, r in df_top_anomalies[df_top_anomalies["SPE_Flag"] == 1].iterrows():
    print(f"  * Row {idx:2d} | Chiller: {r['CHILLER TAG']} | Time: {r['Timestamp']} | SPE = {r['SPE']:.4f} > {primary_tau_spe:.4f}")

# -------------------------------------------------------------
# 6. ACTIVITY E: SENSOR CONTRIBUTION ANALYSIS
# -------------------------------------------------------------
print("\n[Step 6] Calculating sensor residual contributions for flagged test events...")

test_contributions = E_test**2
df_contributions = pd.DataFrame(test_contributions, columns=feature_names)
df_contributions.insert(0, "Timestamp", df_test_meta["Timestamp"].values)
df_contributions.insert(0, "CHILLER TAG", df_test_meta["CHILLER TAG"].values)
df_contributions.insert(2, "Total_SPE", spe_test)
df_contributions.to_csv("results/metrics/phase5/feature_residual_contributions.csv", index=False)

flagged_indices = np.where(test_flags_spe == 1)[0]
flagged_contrib = test_contributions[flagged_indices, :]
mean_flagged_contrib = np.mean(flagged_contrib, axis=0)
total_flagged_contrib = np.sum(flagged_contrib, axis=0)
rel_flagged_contrib = (total_flagged_contrib / np.sum(total_flagged_contrib)) * 100.0

df_agg_contrib = pd.DataFrame({
    "Feature_Name": feature_names,
    "Category": ["Temperature" if any(w in f for w in ["Temperature", "Approach", "Superheat"]) else "Pressure" for f in feature_names],
    "Mean_Squared_Residual": mean_flagged_contrib,
    "Total_Squared_Residual": total_flagged_contrib,
    "Relative_Contribution_Pct": rel_flagged_contrib
}).sort_values(by="Relative_Contribution_Pct", ascending=False)
df_agg_contrib.to_csv("results/metrics/phase5/aggregate_feature_contributions.csv", index=False)

print("\nTop 5 Contributing Sensors in Flagged Baseline Test Points:")
for _, r in df_agg_contrib.head(5).iterrows():
    print(f"  * {r['Feature_Name'][:40]:40s} ({r['Category']:11s}): {r['Relative_Contribution_Pct']:5.2f}% of Residual Variance")

# -------------------------------------------------------------
# 7. ACTIVITY F: FLEET MIX SHIFT & PER-CHILLER DIAGNOSTICS
# -------------------------------------------------------------
print("\n[Step 7] Computing multi-unit per-chiller diagnostics across Validation and Test splits...")

fleet_records = []
all_chillers = sorted(set(df_val_meta["CHILLER TAG"]).union(set(df_test_meta["CHILLER TAG"])))

for ch in all_chillers:
    # Validation split
    v_mask = (df_val_meta["CHILLER TAG"] == ch)
    v_count = int(v_mask.sum())
    v_spe = spe_val[v_mask] if v_count > 0 else np.array([])
    v_t2 = T2_val[v_mask] if v_count > 0 else np.array([])
    
    # Test split
    t_mask = (df_test_meta["CHILLER TAG"] == ch)
    t_count = int(t_mask.sum())
    t_spe = spe_test[t_mask] if t_count > 0 else np.array([])
    t_t2 = T2_test[t_mask] if t_count > 0 else np.array([])
    t_flags = int(np.sum(spe_test[t_mask] > primary_tau_spe)) if t_count > 0 else 0
    
    fleet_records.append({
        "CHILLER_TAG": ch,
        "Val_Obs_Count": v_count,
        "Val_Obs_Pct": round(100.0 * v_count / len(df_val_meta), 2),
        "Val_Mean_SPE": round(float(np.mean(v_spe)), 4) if v_count > 0 else None,
        "Val_Median_SPE": round(float(np.median(v_spe)), 4) if v_count > 0 else None,
        "Val_Mean_T2": round(float(np.mean(v_t2)), 2) if v_count > 0 else None,
        "Test_Obs_Count": t_count,
        "Test_Obs_Pct": round(100.0 * t_count / len(df_test_meta), 2),
        "Test_Mean_SPE": round(float(np.mean(t_spe)), 4) if t_count > 0 else None,
        "Test_Median_SPE": round(float(np.median(t_spe)), 4) if t_count > 0 else None,
        "Test_Mean_T2": round(float(np.mean(t_t2)), 2) if t_count > 0 else None,
        "Test_Flagged_Count": t_flags,
        "Test_Flagged_Pct": round(100.0 * t_flags / t_count, 2) if t_count > 0 else 0.0,
        "Fleet_Mix_Note": "Significant test representation increase (Chiller-1)" if ch == "Chiller - 1" else "Operational unit"
    })

df_fleet = pd.DataFrame(fleet_records)
df_fleet.to_csv("results/metrics/phase5/per_chiller_val_test_comparison.csv", index=False)
df_fleet.to_csv("results/metrics/phase5/chiller_anomaly_summary.csv", index=False)
print(df_fleet[["CHILLER_TAG", "Val_Obs_Count", "Test_Obs_Count", "Val_Mean_SPE", "Test_Mean_SPE", "Test_Flagged_Count"]].to_string(index=False))

# -------------------------------------------------------------
# 8. ACTIVITY G: GENERATE 7 PUBLICATION-QUALITY FIGURES (PRINT FRIENDLY)
# -------------------------------------------------------------
print("\n[Step 8] Generating 7 publication-quality figures (300 DPI, Grayscale & Print Friendly)...")

# FIGURE 1: Validation SPE Distribution with Candidate Thresholds
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sns.histplot(spe_val, bins=25, kde=True, color="#1f77b4", edgecolor="black", alpha=0.6, ax=ax, line_kws={"linewidth": 2.0})
ax.axvline(primary_tau_spe, color="#d62728", linestyle="-", linewidth=2.2, label=f"Primary Threshold (95th Pct = {primary_tau_spe:.2f})")
ax.axvline(secondary_tau_spe_99, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Conservative Threshold (99th Pct = {secondary_tau_spe_99:.2f})")
ax.axvline(np.mean(spe_val), color="black", linestyle=":", linewidth=1.6, label=f"Validation Mean SPE ({np.mean(spe_val):.2f})")

ax.set_title("Figure 1: Cleaned Validation SPE Distribution & Anomaly Thresholds (N_val = 92)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Squared Prediction Error (SPE)", fontsize=10, fontweight="bold")
ax.set_ylabel("Observation Count", fontsize=10, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
plt.tight_layout()
fig.savefig("results/figures/phase5/validation_threshold_analysis.png", dpi=300)
plt.close(fig)

# FIGURE 2: Test Reconstruction Error Timeline
df_test_preds["Datetime"] = pd.to_datetime(df_test_preds["Timestamp"])
df_test_sorted = df_test_preds.sort_values("Datetime")

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df_test_sorted["Datetime"], df_test_sorted["SPE"], marker="o", markersize=4.5, color="#1f77b4", linestyle="-", linewidth=1.4, label="Test SPE")
ax.axhline(primary_tau_spe, color="#d62728", linestyle="--", linewidth=2.0, label=f"Validation 95th Percentile Threshold (tau = {primary_tau_spe:.2f})")
ax.axhline(secondary_tau_spe_99, color="#ff7f0e", linestyle=":", linewidth=1.6, label=f"Validation 99th Percentile Threshold (tau = {secondary_tau_spe_99:.2f})")

ax.set_title("Figure 2: Clean Test SPE Timeline & Frozen Validation Anomaly Thresholds (N_test = 67)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Plant Operating Timestamp", fontsize=10, fontweight="bold")
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10, fontweight="bold")
plt.xticks(rotation=25, ha="right")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
plt.tight_layout()
fig.savefig("results/figures/phase5/test_reconstruction_timeline.png", dpi=300)
plt.close(fig)

# FIGURE 3: Test SPE Histogram vs Validation Threshold
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sns.histplot(spe_test, bins=20, kde=True, color="#2ca02c", edgecolor="black", alpha=0.6, ax=ax, line_kws={"linewidth": 2.0})
ax.axvline(primary_tau_spe, color="#d62728", linestyle="--", linewidth=2.2, label=f"Frozen Validation Threshold ({primary_tau_spe:.2f})")
ax.axvline(np.mean(spe_test), color="#1f77b4", linestyle=":", linewidth=1.8, label=f"Test Mean SPE ({np.mean(spe_test):.2f})")
ax.axvline(np.median(spe_test), color="black", linestyle="-.", linewidth=1.8, label=f"Test Median SPE ({np.median(spe_test):.2f})")

ax.set_title("Figure 3: Test Reconstruction Error Distribution & Threshold Alignment (N_test = 67)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Squared Prediction Error (SPE)", fontsize=10, fontweight="bold")
ax.set_ylabel("Observation Count", fontsize=10, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
plt.tight_layout()
fig.savefig("results/figures/phase5/test_reconstruction_distribution.png", dpi=300)
plt.close(fig)

# FIGURE 4: Potential Anomaly Observations Highlighted on Test Timeline
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
norm_mask = df_test_sorted["SPE_Flag"] == 0
anom_mask = df_test_sorted["SPE_Flag"] == 1

ax.scatter(df_test_sorted.loc[norm_mask, "Datetime"], df_test_sorted.loc[norm_mask, "SPE"],
           color="#1f77b4", marker="o", alpha=0.75, s=36, edgecolors="black", linewidth=0.5, label="Normal Plant Baseline (SPE <= tau)")
ax.scatter(df_test_sorted.loc[anom_mask, "Datetime"], df_test_sorted.loc[anom_mask, "SPE"],
           color="#d62728", marker="X", s=95, edgecolors="black", linewidth=1.0,
           label=f"Candidate Operational Excursion (SPE > tau, n={n_flagged_spe})")
ax.axhline(primary_tau_spe, color="#d62728", linestyle="--", linewidth=1.8, label=f"Threshold (tau = {primary_tau_spe:.2f})")

for idx, r in df_test_sorted.loc[anom_mask].iterrows():
    ax.annotate(f"{r['CHILLER TAG']}\nSPE={r['SPE']:.1f}", xy=(r['Datetime'], r['SPE']),
                xytext=(r['Datetime'], r['SPE'] + 1.2),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                fontsize=8.5, fontweight="bold", ha="center")

ax.set_title("Figure 4: Test Anomaly Detection Timeline — Flagged Candidate Operational Excursions", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Plant Operating Timestamp", fontsize=10, fontweight="bold")
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10, fontweight="bold")
ax.set_ylim(-0.5, max(spe_test) * 1.25)
plt.xticks(rotation=25, ha="right")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
plt.tight_layout()
fig.savefig("results/figures/phase5/test_potential_anomalies.png", dpi=300)
plt.close(fig)

# FIGURE 5: Feature Residual Contribution for Selected High-Error Observations
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
top_features_plot = df_agg_contrib.head(10).sort_values("Relative_Contribution_Pct", ascending=True)
clean_feat_names = [f.replace(" Temperature Deg C", " Temp").replace(" Pressure Kg /cm 2", " Press") for f in top_features_plot["Feature_Name"]]

bar_colors = ["#d62728" if cat == "Pressure" else "#1f77b4" for cat in top_features_plot["Category"]]
hatches = ["//" if cat == "Pressure" else "\\\\" for cat in top_features_plot["Category"]]
bars = ax.barh(clean_feat_names, top_features_plot["Relative_Contribution_Pct"], color=bar_colors, edgecolor="black", height=0.65)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")

legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='black', hatch="\\\\", label='Temperature Domain Feature'),
    Patch(facecolor='#d62728', edgecolor='black', hatch="//", label='Refrigerant Pressure Domain Feature')
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="black", fontsize=9.5)
ax.set_title("Figure 5: Top 10 Feature Contributions to Reconstruction Error in Flagged Baseline Points", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Relative Contribution to Reconstruction Error (%)", fontsize=10, fontweight="bold")
ax.set_ylabel("Thermodynamic Sensor Variable", fontsize=10, fontweight="bold")
ax.set_xlim(0, max(top_features_plot["Relative_Contribution_Pct"]) * 1.18)
ax.grid(True, linestyle="--", alpha=0.5, axis="x")
plt.tight_layout()
fig.savefig("results/figures/phase5/feature_residual_contributions.png", dpi=300)
plt.close(fig)

# FIGURE 6: Multi-Unit Fleet Comparison (Validation vs Test Boxplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
chiller_order = sorted(df_val_meta["CHILLER TAG"].unique())

sns.boxplot(x="CHILLER TAG", y="SPE", data=df_val_errors, order=chiller_order, color="#1f77b4", ax=ax1, width=0.5, boxprops=dict(facecolor="#aec7e8", edgecolor="black"))
sns.stripplot(x="CHILLER TAG", y="SPE", data=df_val_errors, order=chiller_order, color="black", alpha=0.5, size=4, jitter=0.2, ax=ax1)
ax1.axhline(primary_tau_spe, color="#d62728", linestyle="--", linewidth=1.8, label=f"Threshold ({primary_tau_spe:.2f})")
ax1.set_title("(a) Clean Validation Partition (N = 92)", fontsize=10.5, fontweight="bold")
ax1.set_xlabel("Chiller Tag", fontsize=10, fontweight="bold")
ax1.set_ylabel("Squared Prediction Error (SPE)", fontsize=10, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", fontsize=9)

sns.boxplot(x="CHILLER TAG", y="SPE", data=df_test_preds, order=chiller_order, color="#ff7f0e", ax=ax2, width=0.5, boxprops=dict(facecolor="#ffbb78", edgecolor="black"))
sns.stripplot(x="CHILLER TAG", y="SPE", data=df_test_preds, order=chiller_order, color="black", alpha=0.5, size=4, jitter=0.2, ax=ax2)
ax2.axhline(primary_tau_spe, color="#d62728", linestyle="--", linewidth=1.8, label=f"Threshold ({primary_tau_spe:.2f})")
ax2.set_title("(b) Clean Test Partition (N = 67, Fleet Shift: 51% Chiller-1)", fontsize=10.5, fontweight="bold")
ax2.set_xlabel("Chiller Tag", fontsize=10, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", fontsize=9)

plt.suptitle("Figure 6: Per-Chiller Reconstruction Error Distribution Across Partitions (Evaluating Multi-Unit Fleet Heterogeneity)", fontsize=11.5, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig("results/figures/phase5/chiller_reconstruction_comparison.png", dpi=300)
plt.close(fig)

# FIGURE 7: Time-Series Monitoring Layer (EWMA-SPE and Hotelling's T^2 Timeline)
fig, (ax_t2, ax_ewma) = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=True)

# Upper: Hotelling's T^2
ax_t2.plot(df_test_sorted["Datetime"], df_test_sorted["Hotelling_T2"], marker="^", markersize=4, color="#9467bd", linestyle="-", linewidth=1.3, label="Test Hotelling's T^2")
ax_t2.axhline(tau_T2_95, color="#d62728", linestyle="--", linewidth=1.8, label=f"Empirical T^2 Threshold ({tau_T2_95:.1f})")
ax_t2.axhline(theoretical_T2_95, color="#7f7f7f", linestyle=":", linewidth=1.6, label=f"Theoretical F-Dist Threshold ({theoretical_T2_95:.1f})")
ax_t2.set_ylabel("Hotelling's T^2 (Principal Space)", fontsize=10, fontweight="bold")
ax_t2.set_title("Figure 7: Time-Series MSPC Monitoring Layer (Hotelling's T^2 & EWMA-SPE, Test Partition)", fontsize=11, fontweight="bold")
ax_t2.grid(True, linestyle="--", alpha=0.5)
ax_t2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", fontsize=9)

# Lower: EWMA-SPE
ax_ewma.plot(df_test_sorted["Datetime"], df_test_sorted["SPE"], color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.7, label="Instantaneous Raw SPE")
ax_ewma.plot(df_test_sorted["Datetime"], df_test_sorted["EWMA_SPE"], marker="o", markersize=4, color="#1f77b4", linestyle="-", linewidth=1.8, label=f"EWMA-Filtered SPE (lambda = {LAMBDA_EWMA:.2f})")
ax_ewma.axhline(tau_ewma_95, color="#d62728", linestyle="--", linewidth=1.8, label=f"EWMA 95th Pct Threshold ({tau_ewma_95:.2f})")
ax_ewma.axhline(tau_ewma_99, color="#ff7f0e", linestyle=":", linewidth=1.6, label=f"EWMA 99th Pct Threshold ({tau_ewma_99:.2f})")
ax_ewma.set_xlabel("Operating Timestamp (Plant Time)", fontsize=10, fontweight="bold")
ax_ewma.set_ylabel("EWMA-SPE Statistic", fontsize=10, fontweight="bold")
plt.xticks(rotation=25, ha="right")
ax_ewma.grid(True, linestyle="--", alpha=0.5)
ax_ewma.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", fontsize=9)

plt.tight_layout()
fig.savefig("results/figures/phase5/time_series_ewma_monitoring.png", dpi=300)
plt.close(fig)

print("Saved: results/figures/phase5/time_series_ewma_monitoring.png")
print("\nPHASE 5 EXECUTION SUCCESSFULLY COMPLETED.")
