"""
Phase 4: PCA Model Development
==============================
Project: AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Detector

This script executes Phase 4:
- Activity A: Mathematical Foundations & Covariance Formulation
- Activity B: Exploratory Full Spectrum PCA Analysis (Scree, Eigenvalues, Variance)
- Activity C: Candidate Component Selection Comparison (80%, 85%, 90%, 95%, Kaiser)
- Activity D: Final PCA Model Training on Training Set Only (k = 12, 85.46% variance)
- Activity E: Component Loading Analysis (Eigenvectors & Feature Contributions)
- Activity F: Diagnostic Visualizations (Scree, Cumulative Variance, Loadings, Scores, Chiller Diagnostics)

Saves:
- data/processed/pca_model.joblib
- data/processed/pca_configuration.json
- results/metrics/pca_explained_variance.csv
- results/metrics/pca_component_loadings.csv
- results/metrics/pca_component_selection_comparison.csv
- results/metrics/pca_training_scores.csv
- results/figures/pca/scree_plot.png
- results/figures/pca/cumulative_explained_variance.png
- results/figures/pca/component_loadings.png
- results/figures/pca/component_contribution.png
- results/figures/pca/training_component_scores.png
- results/figures/pca/chiller_component_diagnostics.png
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
PROCESSED_DIR = os.path.join("data", "processed")
METRICS_DIR = os.path.join("results", "metrics")
FIGURES_DIR = os.path.join("results", "figures", "pca")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TRAIN_FEAT_PATH = os.path.join(PROCESSED_DIR, "train_features.csv")
TRAIN_META_PATH = os.path.join(PROCESSED_DIR, "train_metadata.csv")


def load_training_data():
    """Load the preprocessed training feature matrix and metadata."""
    if not os.path.exists(TRAIN_FEAT_PATH):
        raise FileNotFoundError(f"Training features not found at {TRAIN_FEAT_PATH}. Run Phase 3 first.")
    
    X_train = pd.read_csv(TRAIN_FEAT_PATH)
    meta_train = pd.read_csv(TRAIN_META_PATH) if os.path.exists(TRAIN_META_PATH) else None
    
    print(f"Loaded training features: {X_train.shape[0]} rows x {X_train.shape[1]} columns")
    return X_train, meta_train


def activity_b_exploratory_pca(X_train):
    """Activity B: Full spectrum PCA analysis on training set only."""
    print("=" * 80)
    print("ACTIVITY B: FULL-SPECTRUM PCA EXPLORATION (TRAINING SET ONLY)")
    print("=" * 80)
    
    # Fit full PCA (all 28 components)
    pca_full = PCA(svd_solver="full", whiten=False)
    pca_full.fit(X_train)
    
    eig_vals = pca_full.explained_variance_
    evr = pca_full.explained_variance_ratio_
    cev = np.cumsum(evr)
    sing_vals = pca_full.singular_values_
    
    var_records = []
    for i in range(len(evr)):
        var_records.append({
            "Principal_Component": f"PC{i+1}",
            "Component_Index": i + 1,
            "Eigenvalue_Lambda": round(eig_vals[i], 6),
            "Singular_Value": round(sing_vals[i], 6),
            "Explained_Variance_Ratio": round(evr[i], 6),
            "Explained_Variance_Pct": round(evr[i] * 100, 2),
            "Cumulative_Explained_Variance_Ratio": round(cev[i], 6),
            "Cumulative_Explained_Variance_Pct": round(cev[i] * 100, 2),
            "Kaiser_Criterion_Lambda_Ge_1": "Pass" if eig_vals[i] >= 1.0 else "Fail"
        })
        
    df_var = pd.DataFrame(var_records)
    var_metrics_path = os.path.join(METRICS_DIR, "pca_explained_variance.csv")
    df_var.to_csv(var_metrics_path, index=False)
    print(f"Saved full explained variance spectrum to: {var_metrics_path}")
    
    return pca_full, df_var


def activity_c_component_selection(df_var, total_features=28):
    """Activity C: Compare candidate component retention thresholds and justify selection."""
    print("\n" + "=" * 80)
    print("ACTIVITY C: COMPONENT SELECTION COMPARISON & TRADE-OFF ANALYSIS")
    print("=" * 80)
    
    candidates = [
        {"Name": "Kaiser Criterion (Lambda >= 1.0)", "Criteria": "Eigenvalue >= 1.0", "Target_Pct": "~70%"},
        {"Name": "Moderate Compression (~80% Variance)", "Criteria": "Cumulative EVR >= 0.80", "Target_Pct": "80%"},
        {"Name": "Recommended Baseline (~85% Variance)", "Criteria": "Cumulative EVR >= 0.85", "Target_Pct": "85%"},
        {"Name": "High Fidelity (~90% Variance)", "Criteria": "Cumulative EVR >= 0.90", "Target_Pct": "90%"},
        {"Name": "Very High Retention (~95% Variance)", "Criteria": "Cumulative EVR >= 0.95", "Target_Pct": "95%"}
    ]
    
    comparison_rows = []
    
    # 1. Kaiser: lambda >= 1.0
    k_kaiser = (df_var["Eigenvalue_Lambda"] >= 1.0).sum()
    row_kaiser = df_var.iloc[k_kaiser - 1]
    comparison_rows.append({
        "Candidate_Strategy": "Kaiser Criterion (Lambda >= 1.0)",
        "Components_Retained_k": k_kaiser,
        "Dimensions_Removed_Residuals": total_features - k_kaiser,
        "Cumulative_Explained_Variance_Pct": row_kaiser["Cumulative_Explained_Variance_Pct"],
        "Residual_Unexplained_Variance_Pct": round(100.0 - row_kaiser["Cumulative_Explained_Variance_Pct"], 2),
        "Detection_Tradeoff_Analysis": "Conservative dimension reduction. Leaves 27.67% variance in residuals; risks higher normal validation SPE noise."
    })
    
    # 2. 80% threshold
    k_80 = (df_var["Cumulative_Explained_Variance_Ratio"] >= 0.80).idxmax() + 1
    row_80 = df_var.iloc[k_80 - 1]
    comparison_rows.append({
        "Candidate_Strategy": "80% Cumulative Variance",
        "Components_Retained_k": k_80,
        "Dimensions_Removed_Residuals": total_features - k_80,
        "Cumulative_Explained_Variance_Pct": row_80["Cumulative_Explained_Variance_Pct"],
        "Residual_Unexplained_Variance_Pct": round(100.0 - row_80["Cumulative_Explained_Variance_Pct"], 2),
        "Detection_Tradeoff_Analysis": "Good compression (k=11). Retains 17 residual dimensions for anomaly projection."
    })
    
    # 3. 85% threshold (RECOMMENDED)
    k_85 = (df_var["Cumulative_Explained_Variance_Ratio"] >= 0.85).idxmax() + 1
    row_85 = df_var.iloc[k_85 - 1]
    comparison_rows.append({
        "Candidate_Strategy": "85% Cumulative Variance (Recommended)",
        "Components_Retained_k": k_85,
        "Dimensions_Removed_Residuals": total_features - k_85,
        "Cumulative_Explained_Variance_Pct": row_85["Cumulative_Explained_Variance_Pct"],
        "Residual_Unexplained_Variance_Pct": round(100.0 - row_85["Cumulative_Explained_Variance_Pct"], 2),
        "Detection_Tradeoff_Analysis": "RECOMMENDED: Captures 85.46% of core thermodynamic covariance while reserving 16 orthogonal residual dimensions to catch decoupled anomaly vectors."
    })
    
    # 4. 90% threshold
    k_90 = (df_var["Cumulative_Explained_Variance_Ratio"] >= 0.90).idxmax() + 1
    row_90 = df_var.iloc[k_90 - 1]
    comparison_rows.append({
        "Candidate_Strategy": "90% Cumulative Variance",
        "Components_Retained_k": k_90,
        "Dimensions_Removed_Residuals": total_features - k_90,
        "Cumulative_Explained_Variance_Pct": row_90["Cumulative_Explained_Variance_Pct"],
        "Residual_Unexplained_Variance_Pct": round(100.0 - row_90["Cumulative_Explained_Variance_Pct"], 2),
        "Detection_Tradeoff_Analysis": "High retention (k=14). Leaves 14 residual dimensions. Good fit for low-noise systems."
    })
    
    # 5. 95% threshold
    k_95 = (df_var["Cumulative_Explained_Variance_Ratio"] >= 0.95).idxmax() + 1
    row_95 = df_var.iloc[k_95 - 1]
    comparison_rows.append({
        "Candidate_Strategy": "95% Cumulative Variance",
        "Components_Retained_k": k_95,
        "Dimensions_Removed_Residuals": total_features - k_95,
        "Cumulative_Explained_Variance_Pct": row_95["Cumulative_Explained_Variance_Pct"],
        "Residual_Unexplained_Variance_Pct": round(100.0 - row_95["Cumulative_Explained_Variance_Pct"], 2),
        "Detection_Tradeoff_Analysis": "High component count (k=17). Only 11 residual dimensions remain; risks reconstructing subtle anomalies inside the model."
    })
    
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(METRICS_DIR, "pca_component_selection_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"Saved component selection comparison to: {comp_path}")
    print("\n--- COMPONENT SELECTION COMPARISON TABLE ---")
    print(df_comp[["Candidate_Strategy", "Components_Retained_k", "Cumulative_Explained_Variance_Pct", "Residual_Unexplained_Variance_Pct"]].to_string(index=False))
    
    return k_85  # Return recommended k=12


def activity_d_final_pca_training(X_train, k=12):
    """Activity D: Fit the final PCA detector on training data with selected k."""
    print("\n" + "=" * 80)
    print(f"ACTIVITY D: FINAL PCA MODEL TRAINING (k = {k} COMPONENTS)")
    print("=" * 80)
    
    pca_model = PCA(n_components=k, svd_solver="full", whiten=False)
    pca_model.fit(X_train)
    
    # Verify explained variance
    evr_k = pca_model.explained_variance_ratio_
    cev_k = np.sum(evr_k)
    print(f"Fitted PCA with k={k} components on {X_train.shape[0]} training observations.")
    print(f"Cumulative Explained Variance: {cev_k * 100:.2f}%")
    print(f"Residual Unexplained Variance: {(1.0 - cev_k) * 100:.2f}%")
    
    # Save model artifact
    model_path = os.path.join(PROCESSED_DIR, "pca_model.joblib")
    joblib.dump(pca_model, model_path)
    print(f"Saved fitted PCA model to: {model_path}")
    
    # Save configuration JSON
    config = {
        "model_type": "Principal Component Analysis (PCA)",
        "assigned_method": "PCA-Based Multivariate Anomaly Detector",
        "student_id": "24BCE0551",
        "n_features_in": int(pca_model.n_features_in_),
        "feature_names": list(X_train.columns),
        "n_training_samples": int(X_train.shape[0]),
        "n_components_retained": int(k),
        "cumulative_explained_variance_ratio": round(float(cev_k), 6),
        "cumulative_explained_variance_pct": round(float(cev_k * 100), 2),
        "residual_unexplained_variance_pct": round(float((1.0 - cev_k) * 100), 2),
        "svd_solver": "full",
        "whiten": False,
        "eigenvalues": [round(float(v), 6) for v in pca_model.explained_variance_],
        "explained_variance_ratios": [round(float(v), 6) for v in pca_model.explained_variance_ratio_],
        "singular_values": [round(float(v), 6) for v in pca_model.singular_values_],
        "model_artifact_path": model_path,
        "selection_rationale": f"Selected k={k} components (~85.46% variance) balancing normal correlation retention against sufficient residual subspace (16 dimensions) for Squared Prediction Error (SPE) anomaly sensitivity."
    }
    
    config_path = os.path.join(PROCESSED_DIR, "pca_configuration.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved PCA configuration metadata to: {config_path}")
    
    return pca_model, config


def activity_e_loading_analysis(pca_model, X_train, k=12):
    """Activity E: Component loading analysis and eigenvector interpretation."""
    print("\n" + "=" * 80)
    print("ACTIVITY E: COMPONENT LOADING & EIGENVECTOR ANALYSIS")
    print("=" * 80)
    
    feature_names = list(X_train.columns)
    loadings = pca_model.components_  # Shape: (k, 28)
    
    loading_records = []
    for f_idx, f_name in enumerate(feature_names):
        cat = "Temperature" if ("temp" in f_name.lower() or "deg c" in f_name.lower() or "approach" in f_name.lower()) else "Pressure"
        row_dict = {
            "Feature_Name": f_name,
            "Category": cat
        }
        for pc_idx in range(k):
            val = loadings[pc_idx, f_idx]
            row_dict[f"PC{pc_idx+1}_Loading"] = round(val, 6)
            row_dict[f"PC{pc_idx+1}_Abs_Loading"] = round(abs(val), 6)
        loading_records.append(row_dict)
        
    df_loadings = pd.DataFrame(loading_records)
    loadings_path = os.path.join(METRICS_DIR, "pca_component_loadings.csv")
    df_loadings.to_csv(loadings_path, index=False)
    print(f"Saved component loadings matrix to: {loadings_path}")
    
    # Compute training scores (projections): Z = X * V^T
    train_scores = pca_model.transform(X_train)
    score_cols = [f"PC{i+1}" for i in range(k)]
    df_scores = pd.DataFrame(train_scores, columns=score_cols)
    scores_path = os.path.join(METRICS_DIR, "pca_training_scores.csv")
    df_scores.to_csv(scores_path, index=False)
    print(f"Saved training component scores to: {scores_path}")
    
    # Identify top contributors for PC1 and PC2
    pc1_top = df_loadings.sort_values(by="PC1_Abs_Loading", ascending=False).head(5)
    pc2_top = df_loadings.sort_values(by="PC2_Abs_Loading", ascending=False).head(5)
    
    print("\nTop 5 Contributing Features for PC1 (Captures 23.77% variance):")
    for _, r in pc1_top.iterrows():
        print(f"  * {r['Feature_Name'][:40]:40s} | Loading = {r['PC1_Loading']:+.4f} | Abs = {r['PC1_Abs_Loading']:.4f}")
        
    print("\nTop 5 Contributing Features for PC2 (Captures 13.64% variance):")
    for _, r in pc2_top.iterrows():
        print(f"  * {r['Feature_Name'][:40]:40s} | Loading = {r['PC2_Loading']:+.4f} | Abs = {r['PC2_Abs_Loading']:.4f}")
        
    return df_loadings, df_scores


def activity_f_visualizations(pca_full, df_var, pca_model, df_loadings, df_scores, meta_train, k=12):
    """Activity F: Generate all 6 diagnostic visualizations at 300 DPI."""
    print("\n" + "=" * 80)
    print("ACTIVITY F: GENERATING DIAGNOSTIC VISUALIZATIONS (300 DPI)")
    print("=" * 80)
    
    # FIGURE 1: Scree Plot (Eigenvalues vs Component Index)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    pcs = range(1, len(df_var) + 1)
    ax.plot(pcs, df_var["Eigenvalue_Lambda"], marker="o", color="#2b5c8f", linewidth=1.8, markersize=5, label="Eigenvalue (λ)")
    ax.axhline(y=1.0, color="#d95f02", linestyle="--", linewidth=1.5, label="Kaiser Criterion (λ = 1.0)")
    ax.axvline(x=k, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Selected Cutoff (k = {k})")
    ax.set_xlabel("Principal Component Index", fontsize=11)
    ax.set_ylabel("Eigenvalue (Explained Variance Magnitude)", fontsize=11)
    ax.set_title("Figure 1: PCA Scree Plot — Eigenvalues across 28 Thermodynamic Variables (Training Set)", fontsize=11.5, pad=12)
    ax.set_xticks(range(1, 29, 2))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(framealpha=0.9, fontsize=9.5)
    
    # Annotate elbow
    ax.annotate("Elbow Transition\n(k ≈ 3 to 5)", xy=(4, df_var.loc[3, "Eigenvalue_Lambda"]), xytext=(7, 4.5),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "scree_plot.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig1_path}")
    
    # FIGURE 2: Cumulative Explained Variance Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(pcs, df_var["Cumulative_Explained_Variance_Pct"], marker="s", color="#7570b3", linewidth=1.8, markersize=5, label="Cumulative Explained Variance (%)")
    ax.axhline(y=85.0, color="#d95f02", linestyle="--", linewidth=1.3, label="85% Target Threshold")
    ax.axhline(y=90.0, color="#e7298a", linestyle="--", linewidth=1.3, label="90% Target Threshold")
    ax.axvline(x=k, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Selected k = {k} ({df_var.loc[k-1, 'Cumulative_Explained_Variance_Pct']:.2f}%)")
    ax.set_xlabel("Number of Retained Principal Components (k)", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
    ax.set_title("Figure 2: Cumulative Explained Variance Curve (Training Partition, N = 358)", fontsize=11.5, pad=12)
    ax.set_xticks(range(1, 29, 2))
    ax.set_ylim(20, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "cumulative_explained_variance.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig2_path}")
    
    # FIGURE 3: Principal Component Loading Heatmap (28 Features x k Components)
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    loadings_matrix = pca_model.components_  # (k, 28)
    heatmap_df = pd.DataFrame(loadings_matrix.T, index=df_loadings["Feature_Name"], columns=[f"PC{i+1}" for i in range(k)])
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0, vmin=-0.4, vmax=0.4, annot=True, fmt=".2f",
                cbar_kws={"label": "Component Loading Value"}, ax=ax, linewidths=0.4, annot_kws={"size": 7})
    ax.set_title(f"Figure 3: Principal Component Loadings Matrix ({k} Retained Components x 28 Features)", fontsize=11.5, pad=12)
    ax.set_xlabel("Principal Components", fontsize=11, labelpad=8)
    ax.set_ylabel("Thermodynamic Sensor Features", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "component_loadings.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig3_path}")
    
    # FIGURE 4: Individual Component Variance Contribution Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    pc_names = [f"PC{i+1}" for i in range(k)]
    ev_pcts = df_var.loc[:k-1, "Explained_Variance_Pct"]
    bars = ax.bar(pc_names, ev_pcts, color="#1f77b4", edgecolor="black", width=0.6)
    ax.set_xlabel("Retained Principal Components", fontsize=11)
    ax.set_ylabel("Explained Variance (%)", fontsize=11)
    ax.set_title(f"Figure 4: Variance Explained by Individual Retained Components (Total = {np.sum(ev_pcts):.2f}%)", fontsize=11.5, pad=12)
    ax.set_ylim(0, max(ev_pcts) * 1.2)
    for bar, pct in zip(bars, ev_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 0.6, f"{pct:.1f}%", ha="center", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "component_contribution.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig4_path}")
    
    # FIGURE 5: Training Component Scores Scatter (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    scatter = ax.scatter(df_scores["PC1"], df_scores["PC2"], color="#2b5c8f", alpha=0.65, s=28, edgecolors="none")
    ax.set_xlabel(f"PC1 Score (Variance Explained: {df_var.loc[0, 'Explained_Variance_Pct']:.1f}%)", fontsize=10.5)
    ax.set_ylabel(f"PC2 Score (Variance Explained: {df_var.loc[1, 'Explained_Variance_Pct']:.1f}%)", fontsize=10.5)
    ax.set_title("Figure 5: Training Observations Projected into Principal Subspace (PC1 vs PC2, N = 358)", fontsize=11.5, pad=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "training_component_scores.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig5_path}")
    
    # FIGURE 6: Chiller-Specific Score Diagnostic (PC1 vs PC2 colored by Chiller Tag)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    if meta_train is not None and "CHILLER TAG" in meta_train.columns:
        chillers = meta_train["CHILLER TAG"].unique()
        palette = sns.color_palette("tab10", n_colors=len(chillers))
        for ch, color in zip(chillers, palette):
            mask = (meta_train["CHILLER TAG"] == ch)
            ax.scatter(df_scores.loc[mask, "PC1"], df_scores.loc[mask, "PC2"], label=ch, color=color, alpha=0.7, s=30, edgecolors="none")
        ax.legend(title="Chiller Unit", framealpha=0.9, fontsize=9)
    else:
        ax.scatter(df_scores["PC1"], df_scores["PC2"], color="#2b5c8f", alpha=0.65, s=28)
        
    ax.set_xlabel(f"PC1 Score ({df_var.loc[0, 'Explained_Variance_Pct']:.1f}%)", fontsize=10.5)
    ax.set_ylabel(f"PC2 Score ({df_var.loc[1, 'Explained_Variance_Pct']:.1f}%)", fontsize=10.5)
    ax.set_title("Figure 6: Chiller-Specific Score Diagnostic (Evaluating Fleet Homogeneity in PC Space)", fontsize=11.5, pad=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "chiller_component_diagnostics.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {fig6_path}")


def main():
    print("=" * 80)
    print("STARTING PHASE 4 PCA MODEL DEVELOPMENT PIPELINE")
    print("=" * 80)
    
    X_train, meta_train = load_training_data()
    pca_full, df_var = activity_b_exploratory_pca(X_train)
    k_selected = activity_c_component_selection(df_var, total_features=X_train.shape[1])
    pca_model, config = activity_d_final_pca_training(X_train, k=k_selected)
    df_loadings, df_scores = activity_e_loading_analysis(pca_model, X_train, k=k_selected)
    activity_f_visualizations(pca_full, df_var, pca_model, df_loadings, df_scores, meta_train, k=k_selected)
    
    print("\n" + "=" * 80)
    print("PHASE 4 PCA MODEL DEVELOPMENT COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()

