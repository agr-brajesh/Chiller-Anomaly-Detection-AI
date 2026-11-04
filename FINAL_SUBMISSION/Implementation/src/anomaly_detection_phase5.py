"""
Phase 5: Reconstruction Error Derivation and PCA-Based Anomaly Detection
Script: src/anomaly_detection_phase5.py
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector

Strict Constraints:
- Reuses fitted imputer.joblib, scaler.joblib, and pca_model.joblib from Phases 3 & 4.
- Zero refitting of transformers or retraining of PCA.
- Threshold derived strictly from Validation data only.
- Test data evaluated using frozen validation threshold.
- No synthetic anomalies in Phase 5.
- Sum of squared residuals (SPE), NOT MSE.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
os.makedirs("results/metrics/phase5", exist_ok=True)
os.makedirs("results/figures/phase5", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD SAVED ARTIFACTS
# -------------------------------------------------------------
print("="*70)
print("PHASE 5: RECONSTRUCTION ERROR & ANOMALY DETECTION PIPELINE")
print("="*70)

print("\n[Step 1] Loading existing Phase 3 and 4 artifacts...")
pca_model_path = "data/processed/pca_model.joblib"
imputer_path = "data/processed/imputer.joblib"
scaler_path = "data/processed/scaler.joblib"
train_feat_path = "data/processed/train_features.csv"
val_feat_path = "data/processed/validation_features.csv"
test_feat_path = "data/processed/test_features.csv"
train_meta_path = "data/processed/train_metadata.csv"
val_meta_path = "data/processed/validation_metadata.csv"
test_meta_path = "data/processed/test_metadata.csv"
config_path = "data/processed/pca_configuration.json"

pca = joblib.load(pca_model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

df_train_feat = pd.read_csv(train_feat_path)
df_val_feat = pd.read_csv(val_feat_path)
df_test_feat = pd.read_csv(test_feat_path)

df_train_meta = pd.read_csv(train_meta_path)
df_val_meta = pd.read_csv(val_meta_path)
df_test_meta = pd.read_csv(test_meta_path)

with open(config_path, "r") as f:
    pca_config = json.load(f)

feature_names = pca_config["feature_names"]
k_components = pca_config["n_components_retained"]
n_features = pca_config["n_features_in"]

print(f"Loaded PCA Model: {k_components} components retained from {n_features} features.")
print(f"Train Shape: {df_train_feat.shape}, Val Shape: {df_val_feat.shape}, Test Shape: {df_test_feat.shape}")
print(f"Feature names count: {len(feature_names)}")

# -------------------------------------------------------------
# 2. ACTIVITY A: PCA TRANSFORMATION AND RECONSTRUCTION
# -------------------------------------------------------------
print("\n[Step 2] Executing PCA projection and inverse reconstruction in standardized space...")

# Verify feature names match
assert list(df_val_feat.columns) == feature_names, "Validation feature columns do not match PCA feature names!"
assert list(df_test_feat.columns) == feature_names, "Test feature columns do not match PCA feature names!"

# Validation data reconstruction
Z_val = df_val_feat.values
Z_val_pca = pca.transform(Z_val)
Z_val_recon = pca.inverse_transform(Z_val_pca)

# Test data reconstruction
Z_test = df_test_feat.values
Z_test_pca = pca.transform(Z_test)
Z_test_recon = pca.inverse_transform(Z_test_pca)

# Train data reconstruction (reference baseline)
Z_train = df_train_feat.values
Z_train_pca = pca.transform(Z_train)
Z_train_recon = pca.inverse_transform(Z_train_pca)

print(f"Validation PCA projection shape: {Z_val_pca.shape}, Reconstruction shape: {Z_val_recon.shape}")
print(f"Test PCA projection shape: {Z_test_pca.shape}, Reconstruction shape: {Z_test_recon.shape}")

# -------------------------------------------------------------
# 3. ACTIVITY B: CALCULATE SQUARED PREDICTION ERROR (SPE)
# -------------------------------------------------------------
print("\n[Step 3] Calculating Squared Prediction Error (SPE) as sum of squared residuals...")

# Residual matrices: E = Z - Z_hat
E_val = Z_val - Z_val_recon
E_test = Z_test - Z_test_recon
E_train = Z_train - Z_train_recon

# SPE_i = sum_j (E_ij^2) across all 28 features (Sum, NOT MSE)
spe_val = np.sum(E_val**2, axis=1)
spe_test = np.sum(E_test**2, axis=1)
spe_train = np.sum(E_train**2, axis=1)

# Save validation reconstruction errors
df_val_errors = df_val_meta.copy()
df_val_errors["SPE"] = spe_val
df_val_errors["MSE_residual"] = spe_val / n_features
df_val_errors.to_csv("results/metrics/phase5/validation_reconstruction_errors.csv", index=False)

# Save test reconstruction errors
df_test_errors = df_test_meta.copy()
df_test_errors["SPE"] = spe_test
df_test_errors["MSE_residual"] = spe_test / n_features
df_test_errors.to_csv("results/metrics/phase5/test_reconstruction_errors.csv", index=False)

# Statistical summaries
def get_spe_stats(spe_array, name):
    return {
        "Dataset": name,
        "N": len(spe_array),
        "Mean": float(np.mean(spe_array)),
        "Std": float(np.std(spe_array)),
        "Median": float(np.median(spe_array)),
        "IQR": float(np.percentile(spe_array, 75) - np.percentile(spe_array, 25)),
        "Min": float(np.min(spe_array)),
        "Max": float(np.max(spe_array)),
        "Q25": float(np.percentile(spe_array, 25)),
        "Q75": float(np.percentile(spe_array, 75)),
        "P90": float(np.percentile(spe_array, 90)),
        "P95": float(np.percentile(spe_array, 95)),
        "P97_5": float(np.percentile(spe_array, 97.5)),
        "P99": float(np.percentile(spe_array, 99))
    }

train_stats = get_spe_stats(spe_train, "Train (Reference)")
val_stats = get_spe_stats(spe_val, "Validation")
test_stats = get_spe_stats(spe_test, "Test")

print(f"Validation SPE Stats: Mean={val_stats['Mean']:.4f}, Median={val_stats['Median']:.4f}, Max={val_stats['Max']:.4f}")
print(f"Test SPE Stats:       Mean={test_stats['Mean']:.4f}, Median={test_stats['Median']:.4f}, Max={test_stats['Max']:.4f}")

# -------------------------------------------------------------
# 4. ACTIVITY C: VALIDATION-BASED ANOMALY THRESHOLDS
# -------------------------------------------------------------
print("\n[Step 4] Evaluating candidate validation thresholds...")

# Candidate percentiles
percentiles = [90.0, 95.0, 97.5, 99.0]
threshold_candidates = []

N_val = len(spe_val)

for p in percentiles:
    t_val = float(np.percentile(spe_val, p))
    # Number of validation observations above threshold
    n_above = int(np.sum(spe_val > t_val))
    pct_above = (n_above / N_val) * 100.0
    # Effective rank in order statistics (1-indexed from max)
    tail_obs_count = int(np.ceil((1.0 - p/100.0) * N_val))
    
    threshold_candidates.append({
        "Percentile": f"{p}%",
        "Threshold_SPE": round(t_val, 4),
        "Val_Observations_Above": n_above,
        "Val_Pct_Above": round(pct_above, 2),
        "Tail_Determining_Points": tail_obs_count,
        "Order_Statistic_Rank": f"Top {tail_obs_count} of {N_val}",
        "Trade_Off_Analysis": (
            "Higher sensitivity; ~5 false alarm candidates under normal operation" if p == 95.0 else
            "Conservative; determined by only ~2 observations" if p == 97.5 else
            "Extreme tail; determined by single outlier observation" if p == 99.0 else
            "Moderate; captures top 10%"
        )
    })

df_threshold_comp = pd.DataFrame(threshold_candidates)
df_threshold_comp.to_csv("results/metrics/phase5/validation_threshold_comparison.csv", index=False)
print(df_threshold_comp.to_string(index=False))

# Selected primary threshold: 95th Percentile
# Rationale: With N_val = 92, the 95th percentile is determined by ~5 observations,
# providing reasonable stability while maintaining conservative industrial sensitivity.
# The 99th percentile is determined by a single observation (1/92 = 1.08%), which is statistically unstable.
primary_percentile = 95.0
selected_threshold = float(np.percentile(spe_val, primary_percentile))

threshold_config = {
    "primary_threshold_percentile": primary_percentile,
    "primary_threshold_spe": round(selected_threshold, 4),
    "primary_threshold_status": "Provisional - Recommended for Student & Instructor Review",
    "validation_sample_size": N_val,
    "rationale": (
        f"With N_val = {N_val}, the 95th percentile ({selected_threshold:.4f}) is anchored by ~5 observations, "
        "providing empirical tail stability. Extreme percentiles (97.5% and 99%) are anchored by only 1-2 points "
        "and exhibit high sample variance."
    ),
    "candidate_thresholds": threshold_candidates
}

with open("results/metrics/phase5/threshold_configuration.json", "w") as f:
    json.dump(threshold_config, f, indent=2)

print(f"\nRecommended Primary Threshold (95th Percentile): SPE = {selected_threshold:.4f}")

# -------------------------------------------------------------
# 5. ACTIVITY D: APPLY FROZEN THRESHOLD TO TEST DATA
# -------------------------------------------------------------
print("\n[Step 5] Applying frozen validation threshold to test partition...")

# Anomaly classification: 1 if SPE > threshold else 0
test_flags = (spe_test > selected_threshold).astype(int)

df_test_preds = df_test_meta.copy()
df_test_preds["SPE"] = spe_test
df_test_preds["Threshold"] = selected_threshold
df_test_preds["Is_Anomaly"] = test_flags
df_test_preds["Flag_Status"] = np.where(test_flags == 1, "Potential Anomaly", "Normal Operation")
df_test_preds.to_csv("results/metrics/phase5/test_anomaly_predictions.csv", index=False)

N_test = len(spe_test)
n_flagged = int(np.sum(test_flags))
n_unflagged = N_test - n_flagged
pct_flagged = (n_flagged / N_test) * 100.0

test_summary = {
    "Total_Test_Observations": N_test,
    "Flagged_Potential_Anomalies": n_flagged,
    "Unflagged_Normal": n_unflagged,
    "Flagged_Percentage": round(pct_flagged, 2),
    "Validation_Threshold_Used": round(selected_threshold, 4),
    "Test_SPE_Mean": round(float(np.mean(spe_test)), 4),
    "Test_SPE_Median": round(float(np.median(spe_test)), 4),
    "Test_SPE_Std": round(float(np.std(spe_test)), 4),
    "Test_SPE_Min": round(float(np.min(spe_test)), 4),
    "Test_SPE_Max": round(float(np.max(spe_test)), 4)
}

pd.DataFrame([test_summary]).to_csv("results/metrics/phase5/test_anomaly_summary.csv", index=False)
print(f"Test Evaluation: Total={N_test}, Flagged={n_flagged} ({pct_flagged:.2f}%), Normal={n_unflagged}")

# Extract top potential anomalies
df_top_anomalies = df_test_preds.sort_values(by="SPE", ascending=False).head(10)
df_top_anomalies.to_csv("results/metrics/phase5/top_potential_anomalies.csv", index=False)

print("\nTop 5 Test Observations with Highest Reconstruction Error:")
print(df_top_anomalies[["CHILLER TAG", "Timestamp", "SPE", "Flag_Status"]].head(5).to_string(index=False))

# -------------------------------------------------------------
# 6. ACTIVITY E: SENSOR CONTRIBUTION ANALYSIS
# -------------------------------------------------------------
print("\n[Step 6] Calculating sensor-wise squared residual contributions...")

# Contribution_ij = E_ij^2 = (Z_ij - Z_hat_ij)^2
test_contributions = E_test**2 # shape (67, 28)

df_contributions = pd.DataFrame(test_contributions, columns=feature_names)
df_contributions.insert(0, "Timestamp", df_test_meta["Timestamp"].values)
df_contributions.insert(0, "CHILLER TAG", df_test_meta["CHILLER TAG"].values)
df_contributions.insert(2, "Total_SPE", spe_test)
df_contributions.to_csv("results/metrics/phase5/feature_residual_contributions.csv", index=False)

# Aggregate feature contributions across all flagged test observations
flagged_indices = np.where(test_flags == 1)[0]
if len(flagged_indices) > 0:
    flagged_contrib = test_contributions[flagged_indices, :]
    mean_flagged_contrib = np.mean(flagged_contrib, axis=0)
    total_flagged_contrib = np.sum(flagged_contrib, axis=0)
    rel_flagged_contrib = (total_flagged_contrib / np.sum(total_flagged_contrib)) * 100.0
else:
    # If none flagged, compute across top 10% SPE
    top_indices = np.argsort(spe_test)[-5:]
    flagged_contrib = test_contributions[top_indices, :]
    mean_flagged_contrib = np.mean(flagged_contrib, axis=0)
    total_flagged_contrib = np.sum(flagged_contrib, axis=0)
    rel_flagged_contrib = (total_flagged_contrib / np.sum(total_flagged_contrib)) * 100.0

df_agg_contrib = pd.DataFrame({
    "Feature_Name": feature_names,
    "Category": ["Temperature" if "Temperature" in f or "Approach" in f or "Superheat" in f else "Pressure" for f in feature_names],
    "Mean_Squared_Residual": mean_flagged_contrib,
    "Total_Squared_Residual": total_flagged_contrib,
    "Relative_Contribution_Pct": rel_flagged_contrib
}).sort_values(by="Relative_Contribution_Pct", ascending=False)

df_agg_contrib.to_csv("results/metrics/phase5/aggregate_feature_contributions.csv", index=False)

print("\nTop 5 Contributing Sensors Across Flagged Test Observations:")
print(df_agg_contrib[["Feature_Name", "Category", "Relative_Contribution_Pct"]].head(5).to_string(index=False))

# -------------------------------------------------------------
# 7. ACTIVITY F: CHILLER-SPECIFIC & TEMPORAL DIAGNOSTICS
# -------------------------------------------------------------
print("\n[Step 7] Computing chiller-specific and temporal diagnostics...")

chiller_stats = []
for chiller_id, group in df_test_preds.groupby("CHILLER TAG"):
    c_n = len(group)
    c_flagged = int(group["Is_Anomaly"].sum())
    c_spe = group["SPE"].values
    chiller_stats.append({
        "CHILLER TAG": chiller_id,
        "Test_Observations": c_n,
        "Flagged_Count": c_flagged,
        "Flagged_Percentage": round((c_flagged / c_n) * 100.0, 2),
        "Mean_SPE": round(float(np.mean(c_spe)), 4),
        "Median_SPE": round(float(np.median(c_spe)), 4),
        "Max_SPE": round(float(np.max(c_spe)), 4),
        "Min_SPE": round(float(np.min(c_spe)), 4)
    })

df_chiller_summary = pd.DataFrame(chiller_stats).sort_values(by="Flagged_Count", ascending=False)
df_chiller_summary.to_csv("results/metrics/phase5/chiller_anomaly_summary.csv", index=False)
print("\nChiller Anomaly Summary on Test Partition:")
print(df_chiller_summary.to_string(index=False))

# -------------------------------------------------------------
# 8. ACTIVITY G: GENERATE 6 REQUIRED PUBLICATION-QUALITY FIGURES
# -------------------------------------------------------------
print("\n[Step 8] Generating 6 high-resolution (300 DPI) publication-quality figures...")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# FIGURE 1: Validation SPE Distribution with Candidate Thresholds
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sns.histplot(spe_val, bins=20, kde=True, color="#2b5c8f", edgecolor="black", alpha=0.65, ax=ax)
colors = {"90%": "#888888", "95%": "#d95f02", "97.5%": "#7570b3", "99%": "#e7298a"}
for tc in threshold_candidates:
    p_name = tc["Percentile"]
    t_val = tc["Threshold_SPE"]
    lw = 2.5 if p_name == "95%" else 1.5
    ls = "-" if p_name == "95%" else "--"
    ax.axvline(t_val, color=colors.get(p_name, "red"), linestyle=ls, linewidth=lw,
               label=f"{p_name} Threshold ({t_val:.2f})")

ax.set_title("Validation Squared Prediction Error (SPE) Distribution & Candidate Thresholds\n(Dataset: Log_Chiller_South, N_val = 92 observations)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Squared Prediction Error (SPE)", fontsize=10)
ax.set_ylabel("Observation Frequency", fontsize=10)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig1_path = "results/figures/phase5/validation_threshold_analysis.png"
fig.savefig(fig1_path)
plt.close(fig)
print(f"Saved: {fig1_path}")

# FIGURE 2: Test Reconstruction Error Timeline
df_test_preds["Datetime"] = pd.to_datetime(df_test_preds["Timestamp"])
df_test_preds_sorted = df_test_preds.sort_values("Datetime")

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df_test_preds_sorted["Datetime"], df_test_preds_sorted["SPE"], marker="o", markersize=4, color="#1f78b4", linewidth=1.2, label="Test SPE")
ax.axhline(selected_threshold, color="#d95f02", linestyle="--", linewidth=2.0, label=f"Validation 95th Percentile Threshold ({selected_threshold:.2f})")

ax.set_title("Test Reconstruction Error Timeline & Validation Anomaly Threshold\n(Test Partition: 2026-03-18 20:00 to 2026-03-20 08:00, N_test = 67)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Operating Timestamp (Plant Time)", fontsize=10)
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10)
plt.xticks(rotation=25, ha="right")
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig2_path = "results/figures/phase5/test_reconstruction_timeline.png"
fig.savefig(fig2_path)
plt.close(fig)
print(f"Saved: {fig2_path}")

# FIGURE 3: Distribution of Test Reconstruction Errors
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sns.histplot(spe_test, bins=15, kde=True, color="#33a02c", edgecolor="black", alpha=0.65, ax=ax)
ax.axvline(selected_threshold, color="#d95f02", linestyle="--", linewidth=2.2, label=f"Validation Threshold ({selected_threshold:.2f})")
ax.axvline(np.mean(spe_test), color="#1f78b4", linestyle=":", linewidth=1.8, label=f"Test Mean SPE ({np.mean(spe_test):.2f})")
ax.axvline(np.median(spe_test), color="#6a3d9a", linestyle="-.", linewidth=1.8, label=f"Test Median SPE ({np.median(spe_test):.2f})")

ax.set_title("Test Reconstruction Error (SPE) Distribution & Threshold Alignment\n(Standardized Feature Space, 28 Variables, 12 Retained PCs)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Squared Prediction Error (SPE)", fontsize=10)
ax.set_ylabel("Count of Observations", fontsize=10)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig3_path = "results/figures/phase5/test_reconstruction_distribution.png"
fig.savefig(fig3_path)
plt.close(fig)
print(f"Saved: {fig3_path}")

# FIGURE 4: Potential Anomaly Observations Highlighted on Test Timeline
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
normal_mask = df_test_preds_sorted["Is_Anomaly"] == 0
anomaly_mask = df_test_preds_sorted["Is_Anomaly"] == 1

ax.scatter(df_test_preds_sorted.loc[normal_mask, "Datetime"], df_test_preds_sorted.loc[normal_mask, "SPE"],
           color="#1f78b4", alpha=0.7, s=35, label="Normal Operation (SPE <= Threshold)")
ax.scatter(df_test_preds_sorted.loc[anomaly_mask, "Datetime"], df_test_preds_sorted.loc[anomaly_mask, "SPE"],
           color="#e31a1c", marker="X", s=80, edgecolors="black", linewidth=0.8,
           label=f"Potential Anomaly (SPE > Threshold, n={n_flagged})")
ax.axhline(selected_threshold, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Threshold ({selected_threshold:.2f})")

ax.set_title("Test Anomaly Detection Timeline: Flagged Operational Discrepancies\n(Discrete Observations, Multi-Chiller Operation)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Operating Timestamp (Plant Time)", fontsize=10)
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10)
plt.xticks(rotation=25, ha="right")
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig4_path = "results/figures/phase5/test_potential_anomalies.png"
fig.savefig(fig4_path)
plt.close(fig)
print(f"Saved: {fig4_path}")

# FIGURE 5: Feature Residual Contribution for Selected High-Error Observations
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
top_features_plot = df_agg_contrib.head(10).sort_values("Relative_Contribution_Pct", ascending=True)

bar_colors = ["#e31a1c" if cat == "Pressure" else "#1f78b4" for cat in top_features_plot["Category"]]
bars = ax.barh(top_features_plot["Feature_Name"], top_features_plot["Relative_Contribution_Pct"], color=bar_colors, edgecolor="black", alpha=0.8)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")

# Custom legend for category
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f78b4', edgecolor='black', label='Temperature Feature'),
    Patch(facecolor='#e31a1c', edgecolor='black', label='Refrigerant Pressure Feature')
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

ax.set_title("Top 10 Feature Contributions to Reconstruction Error in Flagged Test Points\n(Sum of Squared Standardized Residuals: e_ij^2 / SPE_i)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Relative Contribution to Reconstruction Error (%)", fontsize=10)
ax.set_ylabel("Sensor Variable", fontsize=10)
ax.set_xlim(0, max(top_features_plot["Relative_Contribution_Pct"]) * 1.15)
plt.tight_layout()
fig5_path = "results/figures/phase5/feature_residual_contributions.png"
fig.savefig(fig5_path)
plt.close(fig)
print(f"Saved: {fig5_path}")

# FIGURE 6: Test Reconstruction-Error Comparison Across the Five Chillers
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
chiller_order = sorted(df_test_preds["CHILLER TAG"].unique())
sns.boxplot(x="CHILLER TAG", y="SPE", data=df_test_preds, order=chiller_order, palette="Set2", ax=ax, width=0.5, fliersize=4)
sns.stripplot(x="CHILLER TAG", y="SPE", data=df_test_preds, order=chiller_order, color="black", alpha=0.5, size=5, jitter=0.2, ax=ax)
ax.axhline(selected_threshold, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Validation Threshold ({selected_threshold:.2f})")

ax.set_title("Test Reconstruction Error (SPE) Distribution by Chiller Unit\n(Diagnostic Evaluation Across Pooled Fleet)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Chiller Identifier", fontsize=10)
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig6_path = "results/figures/phase5/chiller_reconstruction_comparison.png"
fig.savefig(fig6_path)
plt.close(fig)
print(f"Saved: {fig6_path}")

print("\nAll 6 Phase 5 figures generated successfully.")
print("\nPHASE 5 EXECUTION COMPLETED.")

