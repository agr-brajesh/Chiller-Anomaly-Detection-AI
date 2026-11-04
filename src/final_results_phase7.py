"""
Phase 7: Final Visualization, Results Analysis and Interpretation
Script: src/final_results_phase7.py
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector

Strict Constraints:
- Consolidated analysis using verified artifacts from Phases 1-6.
- Zero retraining of PCA or refitting of transformers.
- Generates 12 publication-quality figures (300 DPI) in results/figures/final/.
- Compiles final_experiment_comparison.csv and research_question_evidence.csv in results/metrics/final/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure output directories exist
os.makedirs("results/figures/final", exist_ok=True)
os.makedirs("results/metrics/final", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

print("="*70)
print("PHASE 7: FINAL VISUALIZATION, RESULTS ANALYSIS & INTERPRETATION")
print("="*70)

# -------------------------------------------------------------
# 1. VERIFY PREVIOUS PHASE OUTPUTS
# -------------------------------------------------------------
print("\n[Step 1] Verifying availability of Phase 1-6 artifacts...")
required_artifacts = [
    # Phase 1 & 2
    "data/raw/official_dataset/Analyse Chillers_AI Course.xls",
    "documentation/initial_data_dictionary.csv",
    "results/metrics/eda/pearson_correlation_matrix.csv",
    "results/metrics/eda/spearman_correlation_matrix.csv",
    # Phase 3
    "data/processed/selected_features.json",
    "data/processed/split_manifest.json",
    "data/processed/train_features.csv",
    "data/processed/validation_features.csv",
    "data/processed/test_features.csv",
    "data/processed/test_metadata.csv",
    "data/processed/scaler.joblib",
    "data/processed/imputer.joblib",
    # Phase 4
    "data/processed/pca_model.joblib",
    "data/processed/pca_configuration.json",
    "results/metrics/pca_explained_variance.csv",
    "results/metrics/pca_component_loadings.csv",
    # Phase 5
    "results/metrics/phase5/threshold_configuration.json",
    "results/metrics/phase5/test_anomaly_predictions.csv",
    "results/metrics/phase5/aggregate_feature_contributions.csv",
    "results/metrics/phase5/chiller_anomaly_summary.csv",
    # Phase 6
    "results/metrics/phase6/anomaly_injection_manifest.csv",
    "results/metrics/phase6/overall_metrics_comparison.csv",
    "results/metrics/phase6/event_detection_results.csv",
    "results/metrics/phase6/clean_vs_synthetic_spe.csv",
    "results/synthetic/spike_test_copy.csv",
    "results/synthetic/drift_test_copy.csv",
    "results/synthetic/relationship_breaking_test_copy.csv"
]

missing = [f for f in required_artifacts if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing required artifacts: {missing}")
print(f"All {len(required_artifacts)} key artifacts verified present on disk.")

# -------------------------------------------------------------
# 2. LOAD KEY ARTIFACTS
# -------------------------------------------------------------
pca = joblib.load("data/processed/pca_model.joblib")
X_test_clean = pd.read_csv("data/processed/test_features.csv")
meta_test = pd.read_csv("data/processed/test_metadata.csv")
meta_test["Datetime"] = pd.to_datetime(meta_test["Timestamp"])

with open("results/metrics/phase5/threshold_configuration.json", "r") as f:
    thresh_cfg = json.load(f)
tau = thresh_cfg["primary_threshold_spe"]

with open("data/processed/pca_configuration.json", "r") as f:
    pca_cfg = json.load(f)

pearson_corr = pd.read_csv("results/metrics/eda/pearson_correlation_matrix.csv", index_col=0)
pca_exp_var = pd.read_csv("results/metrics/pca_explained_variance.csv")
pca_loadings = pd.read_csv("results/metrics/pca_component_loadings.csv", index_col=0)

clean_preds = pd.read_csv("results/metrics/phase5/test_anomaly_predictions.csv")
agg_contrib = pd.read_csv("results/metrics/phase5/aggregate_feature_contributions.csv")

clean_vs_synth = pd.read_csv("results/metrics/phase6/clean_vs_synthetic_spe.csv")
overall_metrics = pd.read_csv("results/metrics/phase6/overall_metrics_comparison.csv")
event_metrics = pd.read_csv("results/metrics/phase6/event_detection_results.csv")
injection_manifest = pd.read_csv("results/metrics/phase6/anomaly_injection_manifest.csv")

# -------------------------------------------------------------
# 3. COMPILE FINAL CONSOLIDATED NUMERICAL COMPARISON CSV
# -------------------------------------------------------------
print("\n[Step 2] Compiling final_experiment_comparison.csv...")
# Combine observation-level and event-level metrics into a consolidated table
consolidated_records = []
for idx, row in overall_metrics.iterrows():
    ev_row = event_metrics[event_metrics["Scenario"] == row["Scenario"]].iloc[0]
    consolidated_records.append({
        "Scenario": row["Scenario"],
        "Total_Observations": row["Total_Obs"],
        "Injected_Observations": row["Positives_True"],
        "True_Positives_TP": row["TP"],
        "False_Positives_FP": row["FP"],
        "False_Negatives_FN": row["FN"],
        "True_Negatives_TN": row["TN"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1_Score": row["F1_Score"],
        "Accuracy": row["Accuracy"],
        "False_Positive_Rate_FPR": row["FPR"],
        "Total_Events_Injected": ev_row["Total_Events"],
        "Events_Detected": ev_row["Detected_Events"],
        "Event_Detection_Rate_Pct": ev_row["Event_Detection_Rate_Pct"],
        "Detection_Delay": ev_row["Detection_Delay"],
        "Clean_Baseline_FP_Contribution": 2,
        "Net_Synthetic_FP": row["FP"] - 2
    })

df_final_comparison = pd.DataFrame(consolidated_records)
df_final_comparison.to_csv("results/metrics/final/final_experiment_comparison.csv", index=False)
print("Saved: results/metrics/final/final_experiment_comparison.csv")

# -------------------------------------------------------------
# 4. COMPILE RESEARCH QUESTION EVIDENCE TABLE CSV
# -------------------------------------------------------------
print("\n[Step 3] Compiling research_question_evidence.csv...")
evidence_records = [
    {
        "Research_Claim": "Strong physical correlation exists between chiller temperature and pressure variables",
        "Actual_Supporting_Result": "Phase 2 correlation analysis confirmed r = +0.9578 (rho = +0.7641) between Discharge Temp Comp 1 and Discharge Pressure Comp 1, and r = +0.8973 between Suction Saturation Temp and Suction Pressure",
        "Limitation": "Linear correlation is strongest within matched thermodynamic loops (high-side lift and low-side evaporation); cross-loop correlations (e.g. approach temp vs EXV diff pressure) are moderate (r ~ 0.3 - 0.5)",
        "Status": "Supported by evidence"
    },
    {
        "Research_Claim": "PCA effectively captures multivariate thermodynamic structure in a reduced subspace",
        "Actual_Supporting_Result": "12 principal components capture 85.46% of variance across 28 features; PC1 (23.77%) captures high-side condenser compression lift and PC2 (13.64%) captures low-side evaporator cooling demand",
        "Limitation": "14.54% variance (16 residual dimensions) is discarded from the principal subspace, which is necessary for SPE detection but discards minor variance",
        "Status": "Supported by evidence"
    },
    {
        "Research_Claim": "Breaking temperature-pressure relationships significantly elevates reconstruction error (SPE)",
        "Actual_Supporting_Result": "In Scenario C, decoupling discharge pressure (-3.5 sigma) while keeping discharge temperature normal drove SPE from clean baseline ~1.65 to 9.54 - 9.74, breaching threshold tau = 9.2931 across all observations",
        "Limitation": "Evaluated on single-variable decoupling of the strongest physical pair; subtle multi-sensor simultaneous drift could exhibit different projection geometry",
        "Status": "Supported by evidence"
    },
    {
        "Research_Claim": "PCA-based detector identifies synthetic temperature and pressure anomalies with high event recall",
        "Actual_Supporting_Result": "100% event detection rate achieved across all three synthetic scenarios (3/3 spikes, 1/1 drift event, 1/1 relational breaking event detected)",
        "Limitation": "In gradual drift, early incipient deviations (< 2.8 sigma) remained below the threshold, resulting in lower observation-level recall (20%) and a 4-hour detection delay",
        "Status": "Supported by evidence"
    },
    {
        "Research_Claim": "Sensor residual contribution isolates the physical loop responsible for anomaly alarms",
        "Actual_Supporting_Result": "Across flagged test points, suction and lift temperatures accounted for over 73% of reconstruction error, correctly pinpointing low-side evaporator loop deviations",
        "Limitation": "Contribution plots indicate the variables most discrepant from model expectation, not necessarily the root-cause physical component failure",
        "Status": "Supported by evidence"
    },
    {
        "Research_Claim": "Multi-chiller pooling under dual-compressor operation provides a stable, generalizable baseline",
        "Actual_Supporting_Result": "Chiller-1 exhibited 0 false alarms across 34 test observations; median SPE is consistent across chillers (1.37 to 2.24)",
        "Limitation": "Chillers 4 and 5 had higher operational variance in test partition, accounting for the 2 natural baseline flags; slight inter-unit baseline shifts exist",
        "Status": "Partially supported"
    },
    {
        "Research_Claim": "Synthetic anomaly detection proves real-world physical chiller fault diagnosis",
        "Actual_Supporting_Result": "Synthetic perturbations were detected with high precision and recall under controlled testing",
        "Limitation": "No verified maintenance logs or real physical fault labels exist in the official operational dataset; real faults may have complex non-linear dynamics",
        "Status": "Cannot be determined"
    }
]

df_rq_evidence = pd.DataFrame(evidence_records)
df_rq_evidence.to_csv("results/metrics/final/research_question_evidence.csv", index=False)
print("Saved: results/metrics/final/research_question_evidence.csv")

# -------------------------------------------------------------
# 5. GENERATE THE 12 FINAL PUBLICATION-QUALITY FIGURES (300 DPI)
# -------------------------------------------------------------
print("\n[Step 4] Generating the 12 final publication-quality figures (300 DPI)...")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# FIGURE 1: Final Temperature-Pressure Correlation Matrix
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
# Subselect 10 key temperature and pressure variables for maximum readability
key_vars = [
    "Evaporator Water In Temperature Deg C",
    "Evaporator Water Out Temperature Deg C",
    "Condenser Water In Temperature Deg C",
    "Condenser Water Out Temperature Deg C",
    "Discharge Temperature Deg C Comp 1",
    "Suction Temperature Deg C",
    "Suction Saturation Temperature Deg C",
    "Discharge Pressure Kg /cm 2 Comp 1",
    "Suction Pressure Kg /cm 2",
    "Chiller Lift Pressure Kg /cm 2 Comp 1"
]
sub_corr = pearson_corr.loc[key_vars, key_vars]
mask = np.triu(np.ones_like(sub_corr, dtype=bool))
sns.heatmap(sub_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"}, ax=ax)
ax.set_title("Figure 1: Core Temperature-Pressure Thermodynamic Correlation Matrix\n(Highlighting Physical Coupling: R134a Vapor-Pressure Thermodynamics)", fontsize=11, fontweight="bold", pad=12)
plt.xticks(rotation=40, ha="right", fontsize=8.5)
plt.yticks(rotation=0, fontsize=8.5)
plt.tight_layout()
fig.savefig("results/figures/final/final_temperature_pressure_correlation.png")
plt.close(fig)
print("Saved: results/figures/final/final_temperature_pressure_correlation.png")

# FIGURE 2: Final PCA Explained Variance & Scree
fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
k_range = pca_exp_var["Component_Index"].values
evr = pca_exp_var["Explained_Variance_Pct"].values
cum_evr = pca_exp_var["Cumulative_Explained_Variance_Pct"].values

color1 = "#1f78b4"
color2 = "#33a02c"

ax1.bar(k_range, evr, color=color1, alpha=0.7, edgecolor="black", width=0.6, label="Individual EVR (%)")
ax1.set_xlabel("Principal Component Number", fontsize=10)
ax1.set_ylabel("Individual Explained Variance (%)", color=color1, fontsize=10)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(0, 29)
ax1.set_ylim(0, 30)

ax2 = ax1.twinx()
ax2.plot(k_range, cum_evr, color=color2, marker="o", markersize=4, linewidth=2, label="Cumulative EVR (%)")
ax2.axhline(85.46, color="#d95f02", linestyle="--", linewidth=1.8, label="Retained Model (k=12: 85.46%)")
ax2.axvline(12, color="#d95f02", linestyle=":", linewidth=1.8)
ax2.set_ylabel("Cumulative Explained Variance (%)", color=color2, fontsize=10)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 105)
ax2.grid(False)

plt.title("Figure 2: PCA Scree Plot and Cumulative Explained Variance Spectrum\n(Selecting k = 12 PCs: 85.46% Retained Variance, 16 Residual Dimensions)", fontsize=11, fontweight="bold", pad=12)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
plt.tight_layout()
fig.savefig("results/figures/final/final_pca_explained_variance.png")
plt.close(fig)
print("Saved: results/figures/final/final_pca_explained_variance.png")

# FIGURE 3: Final PCA Component Loadings Heatmap
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
# Select PC1 to PC5 loading columns specifically
loading_cols = [f"PC{i}_Loading" for i in range(1, 6)]
loadings_sub = pca_loadings[loading_cols].copy()
loadings_sub.columns = [f"PC{i}" for i in range(1, 6)]
sns.heatmap(loadings_sub, annot=True, fmt=".2f", cmap="vlag", vmin=-0.4, vmax=0.4,
            linewidths=0.4, cbar_kws={"label": "Component Eigenvector Loading"}, ax=ax)
ax.set_title("Figure 3: Principal Component Eigenvector Loadings (PC1 to PC5)\n(PC1: High-Side Lift Dynamics | PC2: Low-Side Evaporator Cooling Load)", fontsize=11, fontweight="bold", pad=12)
plt.xticks(fontsize=9, fontweight="bold")
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig("results/figures/final/final_pca_component_loadings.png")
plt.close(fig)
print("Saved: results/figures/final/final_pca_component_loadings.png")

# FIGURE 4: Final Clean Test Reconstruction Error Timeline
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
clean_preds_sorted = clean_preds.sort_values("Timestamp")
clean_preds_sorted["Datetime"] = pd.to_datetime(clean_preds_sorted["Timestamp"])
norm_pts = clean_preds_sorted[clean_preds_sorted["Is_Anomaly"] == 0]
anom_pts = clean_preds_sorted[clean_preds_sorted["Is_Anomaly"] == 1]

ax.plot(clean_preds_sorted["Datetime"], clean_preds_sorted["SPE"], color="#1f78b4", alpha=0.4, linewidth=1.2)
ax.scatter(norm_pts["Datetime"], norm_pts["SPE"], color="#1f78b4", s=35, label=f"Normal Operation (SPE <= tau, n=65)")
ax.scatter(anom_pts["Datetime"], anom_pts["SPE"], color="#e31a1c", marker="X", s=90, edgecolors="black", label=f"Natural Baseline Excursion (SPE > tau, n=2)")
ax.axhline(tau, color="#d95f02", linestyle="--", linewidth=2.0, label=f"Frozen Validation Threshold (tau = {tau:.2f})")
ax.set_title("Figure 4: Clean Test Reconstruction Error (SPE) Timeline\n(Held-Out Test Partition: 2026-03-18 20:00 to 2026-03-20 08:00, N=67)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Plant Operating Timestamp", fontsize=10)
ax.set_ylabel("Squared Prediction Error (SPE)", fontsize=10)
plt.xticks(rotation=20, ha="right")
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig.savefig("results/figures/final/final_clean_test_reconstruction_error.png")
plt.close(fig)
print("Saved: results/figures/final/final_clean_test_reconstruction_error.png")

# FIGURE 5: Final Clean Test Anomaly Distribution
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
sns.histplot(clean_preds["SPE"], bins=18, kde=True, color="#33a02c", edgecolor="black", alpha=0.65, ax=ax)
ax.axvline(tau, color="#d95f02", linestyle="--", linewidth=2.0, label=f"Validation Threshold ({tau:.2f})")
ax.axvline(clean_preds["SPE"].median(), color="#6a3d9a", linestyle="-.", linewidth=1.8, label=f"Median SPE ({clean_preds['SPE'].median():.2f})")
ax.axvline(clean_preds["SPE"].mean(), color="#1f78b4", linestyle=":", linewidth=1.8, label=f"Mean SPE ({clean_preds['SPE'].mean():.2f})")
ax.set_title("Figure 5: Distribution of Test Reconstruction Errors (SPE)\n(Showing Separation of Normal Operation from Baseline Tail Excursions)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Squared Prediction Error (SPE)", fontsize=10)
ax.set_ylabel("Observation Frequency", fontsize=10)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
plt.tight_layout()
fig.savefig("results/figures/final/final_clean_test_anomaly_distribution.png")
plt.close(fig)
print("Saved: results/figures/final/final_clean_test_anomaly_distribution.png")

# FIGURE 6: Final Feature Residual Contributions
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
top_contrib_plot = agg_contrib.head(8).sort_values("Relative_Contribution_Pct", ascending=True)
bar_colors = ["#e31a1c" if cat == "Pressure" else "#1f78b4" for cat in top_contrib_plot["Category"]]
bars = ax.barh(top_contrib_plot["Feature_Name"], top_contrib_plot["Relative_Contribution_Pct"], color=bar_colors, edgecolor="black", alpha=0.85)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f78b4', edgecolor='black', label='Temperature Feature'),
    Patch(facecolor='#e31a1c', edgecolor='black', label='Pressure Feature')
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
ax.set_title("Figure 6: Top Feature Contributions to Reconstruction Error in Flagged Points\n(Sum of Squared Residuals e_ij^2 / SPE_i; Evaporator Loop Drives > 73%)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Relative Residual Contribution (%)", fontsize=10)
ax.set_xlim(0, max(top_contrib_plot["Relative_Contribution_Pct"]) * 1.18)
plt.tight_layout()
fig.savefig("results/figures/final/final_feature_residual_contributions.png")
plt.close(fig)
print("Saved: results/figures/final/final_feature_residual_contributions.png")

# FIGURE 7: Final Synthetic Spike Analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=300, sharex=True)
ax1.plot(meta_test["Datetime"], X_test_clean["Discharge Temperature Deg C Comp 1"], color="#1f78b4", marker="o", markersize=3, label="Clean Standardized Temperature")
X_sp = pd.read_csv("results/synthetic/spike_test_copy.csv")
ax1.scatter([meta_test.loc[18, "Datetime"]], [X_sp.loc[18, "Discharge Temperature Deg C Comp 1"]], color="#e31a1c", marker="^", s=90, zorder=5, label="Synthetic Spike (+4.0 sigma)")
ax1.set_ylabel("Standardized Sensor (z)", fontsize=9.5)
ax1.set_title("Figure 7: Scenario A — Sudden Sensor Spike in Discharge Temperature\n(Immediate Threshold Breach without Pre-Alarm Distortion)", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

ax2.plot(meta_test["Datetime"], clean_vs_synth["Clean_SPE"], color="gray", linestyle=":", label="Clean Baseline SPE")
ax2.plot(meta_test["Datetime"], clean_vs_synth["Spike_SPE"], color="#e31a1c", linewidth=1.2, label="Spike Test SPE")
ax2.scatter(meta_test.loc[[18, 26, 48], "Datetime"], clean_vs_synth.loc[[18, 26, 48], "Spike_SPE"], color="#e31a1c", marker="X", s=80, zorder=5, label="Detected Spike Alarms (SPE > tau)")
ax2.axhline(tau, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Threshold ({tau:.2f})")
ax2.set_xlabel("Timestamp", fontsize=9.5)
ax2.set_ylabel("Reconstruction SPE", fontsize=9.5)
plt.xticks(rotation=20, ha="right")
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
plt.tight_layout()
fig.savefig("results/figures/final/final_synthetic_spike_analysis.png")
plt.close(fig)
print("Saved: results/figures/final/final_synthetic_spike_analysis.png")

# FIGURE 8: Final Synthetic Drift Analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=300, sharex=True)
X_dr = pd.read_csv("results/synthetic/drift_test_copy.csv")
drift_idx = [23, 25, 27, 29, 30]
ax1.plot(meta_test["Datetime"], X_test_clean["Evaporator Water Out Temperature Deg C"], color="#1f78b4", marker="o", markersize=3, label="Clean Standardized Temperature")
ax1.plot(meta_test.loc[drift_idx, "Datetime"], X_dr.loc[drift_idx, "Evaporator Water Out Temperature Deg C"], color="#ff7f00", marker="s", markersize=6, linewidth=1.8, label="Linear Drift (+0.7 to +3.5 sigma)")
ax1.set_ylabel("Standardized Sensor (z)", fontsize=9.5)
ax1.set_title("Figure 8: Scenario B — Gradual Sensor Drift in Evaporator Water Out Temperature\n(Demonstrating Incipient Phase Tolerance and 4-Hour Detection Latency)", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

ax2.plot(meta_test["Datetime"], clean_vs_synth["Clean_SPE"], color="gray", linestyle=":", label="Clean Baseline SPE")
ax2.plot(meta_test["Datetime"], clean_vs_synth["Drift_SPE"], color="#ff7f00", linewidth=1.4, label="Drift Test SPE")
ax2.scatter([meta_test.loc[30, "Datetime"]], [clean_vs_synth.loc[30, "Drift_SPE"]], color="red", marker="X", s=110, zorder=5, label="Alarm Fired at Hour 5 (SPE = 13.50 > tau)")
ax2.axhline(tau, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Threshold ({tau:.2f})")
ax2.set_xlabel("Timestamp", fontsize=9.5)
ax2.set_ylabel("Reconstruction SPE", fontsize=9.5)
plt.xticks(rotation=20, ha="right")
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
plt.tight_layout()
fig.savefig("results/figures/final/final_synthetic_drift_analysis.png")
plt.close(fig)
print("Saved: results/figures/final/final_synthetic_drift_analysis.png")

# FIGURE 9: Final Relationship Breaking Analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=300, sharex=True)
rel_idx = [61, 63, 65]
X_re = pd.read_csv("results/synthetic/relationship_breaking_test_copy.csv")
ax1.plot(meta_test["Datetime"], X_test_clean["Discharge Temperature Deg C Comp 1"], color="#e31a1c", label="Discharge Temp Comp 1 (Normal, Untouched)")
ax1.plot(meta_test["Datetime"], X_test_clean["Discharge Pressure Kg /cm 2 Comp 1"], color="#1f78b4", linestyle="--", label="Discharge Press Comp 1 (Clean Baseline)")
ax1.plot(meta_test.loc[rel_idx, "Datetime"], X_re.loc[rel_idx, "Discharge Pressure Kg /cm 2 Comp 1"], color="#6a3d9a", marker="D", linewidth=1.8, label="Decoupled Discharge Press (-3.5 sigma)")
ax1.set_ylabel("Standardized Sensor (z)", fontsize=9.5)
ax1.set_title("Figure 9: Scenario C — Breaking Correlated Temperature-Pressure Thermodynamics\n(Decoupling r = +0.9578 Physical Relationship Forces Data into Residual Subspace)", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

ax2.plot(meta_test["Datetime"], clean_vs_synth["Clean_SPE"], color="gray", linestyle=":", label="Clean Baseline SPE")
ax2.plot(meta_test["Datetime"], clean_vs_synth["Rel_Break_SPE"], color="#6a3d9a", linewidth=1.4, label="Decoupled Test SPE")
ax2.scatter(meta_test.loc[rel_idx, "Datetime"], clean_vs_synth.loc[rel_idx, "Rel_Break_SPE"], color="red", marker="X", s=80, zorder=5, label="Detected Relational Anomalies (100% Recall)")
ax2.axhline(tau, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Threshold ({tau:.2f})")
ax2.set_xlabel("Timestamp", fontsize=9.5)
ax2.set_ylabel("Reconstruction SPE", fontsize=9.5)
plt.xticks(rotation=20, ha="right")
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
plt.tight_layout()
fig.savefig("results/figures/final/final_relationship_breaking_analysis.png")
plt.close(fig)
print("Saved: results/figures/final/final_relationship_breaking_analysis.png")

# FIGURE 10: Final Confusion Matrices Comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)
gt_spike = pd.read_csv("results/synthetic/spike_ground_truth.csv")["Synthetic_Label"]
gt_drift = pd.read_csv("results/synthetic/drift_ground_truth.csv")["Synthetic_Label"]
gt_rel = pd.read_csv("results/synthetic/relationship_breaking_ground_truth.csv")["Synthetic_Label"]

cms = [
    ("Scenario A: Sudden Spikes", gt_spike, clean_vs_synth["Spike_Flag"]),
    ("Scenario B: Gradual Drift", gt_drift, clean_vs_synth["Drift_Flag"]),
    ("Scenario C: Rel Breaking", gt_rel, clean_vs_synth["Rel_Break_Flag"])
]

for ax, (title, yt, yp) in zip(axes, cms):
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Predicted Label (0=Normal, 1=Anomaly)", fontsize=9)
    ax.set_ylabel("Ground Truth Label (0=Clean, 1=Injected)", fontsize=9)
    ax.set_xticklabels(["Normal (0)", "Anomaly (1)"])
    ax.set_yticklabels(["Unmodified (0)", "Injected (1)"])

plt.suptitle("Figure 10: Confusion Matrices Across the Three Synthetic Anomaly Scenarios\n(Note: FP=2 across all matrices represents the 2 naturally occurring baseline excursions)", fontsize=11, fontweight="bold", y=1.03)
plt.tight_layout()
fig.savefig("results/figures/final/final_confusion_matrices.png", bbox_inches='tight')
plt.close(fig)
print("Saved: results/figures/final/final_confusion_matrices.png")

# FIGURE 11: Final Observation-Level Performance Comparison
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
metrics_cols = ["Precision", "Recall", "F1_Score", "Accuracy"]
df_p = df_final_comparison.set_index("Scenario")[metrics_cols]
df_p.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black", width=0.6)
ax.set_title("Figure 11: Observation-Level Performance Metrics Comparison\n(High Precision & Recall in Spikes and Relationship Breaking; Incipient Trade-Off in Drift)", fontsize=11, fontweight="bold", pad=12)
ax.set_ylabel("Metric Value (0.0 to 1.0)", fontsize=10)
ax.set_xlabel("Synthetic Evaluation Scenario", fontsize=10)
plt.xticks(rotation=15, ha="right")
ax.set_ylim(0, 1.15)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="upper right")
plt.tight_layout()
fig.savefig("results/figures/final/final_performance_comparison.png")
plt.close(fig)
print("Saved: results/figures/final/final_performance_comparison.png")

# FIGURE 12: Final Event Detection Rate Comparison
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
scen_labels = ["Sudden Spikes\n(Scenario A)", "Gradual Drift\n(Scenario B)", "Relationship Breaking\n(Scenario C)"]
rates = df_final_comparison["Event_Detection_Rate_Pct"]
bars = ax.bar(scen_labels, rates, color=["#1f78b4", "#ff7f00", "#6a3d9a"], edgecolor="black", width=0.45)
ax.set_title("Figure 12: Event-Level Detection Rate Across Synthetic Anomaly Categories\n(100% Event Detection: All Operational Distortions Successfully Caught)", fontsize=11, fontweight="bold", pad=12)
ax.set_ylabel("Event Detection Rate (%)", fontsize=10)
ax.set_ylim(0, 120)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 3, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig("results/figures/final/final_event_detection_comparison.png")
plt.close(fig)
print("Saved: results/figures/final/final_event_detection_comparison.png")

print("\nAll 12 final figures generated successfully at 300 DPI.")
print("PHASE 7 ANALYSIS SCRIPT COMPLETED.")
