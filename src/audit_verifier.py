import os
import json
import hashlib
import joblib
import pandas as pd
import numpy as np

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

print("="*60)
print("AUDIT VERIFIER STARTING (READ-ONLY)")
print("="*60)

# 1. Dataset Verification
raw_official = r"data/raw/official_dataset/Analyse Chillers_AI Course.xls"
raw_root = r"Analyse Chillers_AI Course.xls"

official_exists = os.path.exists(raw_official)
root_exists = os.path.exists(raw_root)
official_sha = file_sha256(raw_official) if official_exists else None
root_sha = file_sha256(raw_root) if root_exists else None
official_size = os.path.getsize(raw_official) if official_exists else None
root_size = os.path.getsize(raw_root) if root_exists else None

print(f"Official Dataset: exists={official_exists}, size={official_size}, sha256={official_sha}")
print(f"Root Dataset:     exists={root_exists}, size={root_size}, sha256={root_sha}")
print(f"Identical checksums: {official_sha == root_sha}")

# Read excel sheet names and dimensions
xl = pd.ExcelFile(raw_official)
print(f"Excel Sheet Names ({len(xl.sheet_names)}): {xl.sheet_names}")
for sname in xl.sheet_names:
    df_s = pd.read_excel(xl, sheet_name=sname, header=None)
    print(f"  Sheet '{sname}': shape={df_s.shape}")

# 2. Phase 3 Processed Data Verification
train_feat = pd.read_csv("data/processed/train_features.csv")
val_feat = pd.read_csv("data/processed/validation_features.csv")
test_feat = pd.read_csv("data/processed/test_features.csv")

train_meta = pd.read_csv("data/processed/train_metadata.csv")
val_meta = pd.read_csv("data/processed/validation_metadata.csv")
test_meta = pd.read_csv("data/processed/test_metadata.csv")

print(f"\nTrain Features Shape: {train_feat.shape}, Metadata: {train_meta.shape}")
print(f"Val Features Shape:   {val_feat.shape}, Metadata: {val_meta.shape}")
print(f"Test Features Shape:  {test_feat.shape}, Metadata: {test_meta.shape}")

# Verify temporal split boundaries and no overlap
train_ts = pd.to_datetime(train_meta['Timestamp'])
val_ts = pd.to_datetime(val_meta['Timestamp'])
test_ts = pd.to_datetime(test_meta['Timestamp'])

print(f"Train Timestamps: min={train_ts.min()}, max={train_ts.max()}, unique={train_ts.nunique()}")
print(f"Val Timestamps:   min={val_ts.min()}, max={val_ts.max()}, unique={val_ts.nunique()}")
print(f"Test Timestamps:  min={test_ts.min()}, max={test_ts.max()}, unique={test_ts.nunique()}")

overlap_train_val = set(train_ts).intersection(set(val_ts))
overlap_val_test = set(val_ts).intersection(set(test_ts))
overlap_train_test = set(train_ts).intersection(set(test_ts))
print(f"Overlap train-val: {len(overlap_train_val)}, val-test: {len(overlap_val_test)}, train-test: {len(overlap_train_test)}")

# Verify Chiller representation
print("\nMetadata columns:", train_meta.columns.tolist())
chiller_col = [c for c in train_meta.columns if 'chiller' in c.lower() or 'id' in c.lower()][0]
print(f"Chiller column detected: '{chiller_col}'")
print("Train:\n", train_meta[chiller_col].value_counts())
print("Val:\n", val_meta[chiller_col].value_counts())
print("Test:\n", test_meta[chiller_col].value_counts())

# 3. Transformers Verification
imputer = joblib.load("data/processed/imputer.joblib")
scaler = joblib.load("data/processed/scaler.joblib")
with open("data/processed/selected_features.json", "r") as f:
    sel_features = json.load(f)
with open("data/processed/split_manifest.json", "r") as f:
    split_manifest = json.load(f)

print(f"\nSelected features count: {sel_features['total_selected_features']}")
print(f"Feature list count: {len(sel_features['feature_list'])}")
print(f"Imputer statistics shape: {imputer.statistics_.shape}")
print(f"Scaler mean shape: {scaler.mean_.shape}, scale shape: {scaler.scale_.shape}")

# Verify train feature standardization (mean ~ 0, std ~ 1)
means = train_feat.mean()
stds = train_feat.std(ddof=0)
print(f"Train standardized features max absolute mean: {np.abs(means).max():.6e}")
print(f"Train standardized features min std: {stds.min():.6f}, max std: {stds.max():.6f}")

# 4. PCA Model Verification
pca = joblib.load("data/processed/pca_model.joblib")
with open("data/processed/pca_configuration.json", "r") as f:
    pca_config = json.load(f)

print(f"\nPCA Model: n_components={pca.n_components_}, n_features={pca.n_features_in_}")
print(f"PCA Config: {pca_config['n_components_retained']} components, explained variance: {pca_config['cumulative_explained_variance_pct']}%")

# Test transform and inverse_transform
X_train_pca = pca.transform(train_feat.values)
X_train_recon = pca.inverse_transform(X_train_pca)
recon_err = np.mean((train_feat.values - X_train_recon)**2)
print(f"PCA transform output shape: {X_train_pca.shape}")
print(f"PCA inverse_transform output shape: {X_train_recon.shape}")
print(f"Train reconstruction MSE (residual error): {recon_err:.6f}")

# Check component orthogonality
# Components matrix V has shape (k, p). V @ V.T should be identity matrix (k x k)
V = pca.components_
ortho_check = V @ V.T
ortho_error = np.max(np.abs(ortho_check - np.eye(pca.n_components_)))
print(f"Component orthogonality max absolute deviation from identity: {ortho_error:.6e}")

# Check explained variance ratio
var_ratio = pca.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)
print(f"Component 1 explained variance: {var_ratio[0]*100:.2f}%")
print(f"Component 2 explained variance: {var_ratio[1]*100:.2f}%")
print(f"Component 12 cumulative variance: {cum_var[11]*100:.2f}%")

# Check singular values
print(f"Singular values: {pca.singular_values_[:5]} ... {pca.singular_values_[-1]}")

# 5. File Inventory & Size Check
expected_files = [
    # Documentation
    "documentation/dataset_profile_working.md",
    "documentation/initial_data_dictionary.csv",
    "documentation/phase1_file_inventory.csv",
    "documentation/phase1_decision_log.md",
    "documentation/phase2_eda_working.md",
    "documentation/phase3_preprocessing_working.md",
    "documentation/phase4_pca_working.md",
    "documentation/ai_assistance_log.md",
    "README.md",
    # Data Processed
    "data/processed/selected_features.json",
    "data/processed/split_manifest.json",
    "data/processed/imputer.joblib",
    "data/processed/scaler.joblib",
    "data/processed/pca_model.joblib",
    "data/processed/pca_configuration.json",
    "data/processed/train_features.csv",
    "data/processed/train_metadata.csv",
    "data/processed/validation_features.csv",
    "data/processed/validation_metadata.csv",
    "data/processed/test_features.csv",
    "data/processed/test_metadata.csv",
    # Metrics EDA
    "results/metrics/eda/missing_values_summary.csv",
    "results/metrics/eda/duplicate_analysis.csv",
    "results/metrics/eda/timestamp_continuity_summary.csv",
    "results/metrics/eda/irregular_timestamp_intervals.csv",
    "results/metrics/eda/descriptive_statistics.csv",
    "results/metrics/eda/sensor_quality_summary.csv",
    "results/metrics/eda/zero_variance_features.csv",
    "results/metrics/eda/pearson_correlation_matrix.csv",
    "results/metrics/eda/spearman_correlation_matrix.csv",
    "results/metrics/eda/temperature_pressure_pairwise_correlations.csv",
    # Metrics Preprocessing
    "results/metrics/preprocessing/imputation_statistics.csv",
    "results/metrics/preprocessing/training_feature_variances.csv",
    "results/metrics/preprocessing/scaling_parameters.csv",
    "results/metrics/preprocessing/preprocessing_validation.csv",
    # Metrics PCA
    "results/metrics/pca_explained_variance.csv",
    "results/metrics/pca_component_selection_comparison.csv",
    "results/metrics/pca_component_loadings.csv",
    "results/metrics/pca_training_scores.csv",
    # Figures EDA
    "results/figures/eda/missing_values.png",
    "results/figures/eda/timestamp_continuity.png",
    "results/figures/eda/sensor_distributions.png",
    "results/figures/eda/zero_variance_analysis.png",
    "results/figures/eda/temperature_pressure_heatmap.png",
    "results/figures/eda/temperature_pressure_scatterplots.png",
    "results/figures/eda/sensor_time_series_profiles.png",
    # Figures Preprocessing
    "results/figures/preprocessing/train_validation_test_split.png",
    "results/figures/preprocessing/scaling_comparison.png",
    # Figures PCA
    "results/figures/pca/scree_plot.png",
    "results/figures/pca/cumulative_explained_variance.png",
    "results/figures/pca/component_loadings.png",
    "results/figures/pca/component_contribution.png",
    "results/figures/pca/training_component_scores.png",
    "results/figures/pca/chiller_component_diagnostics.png",
    # Notebook & Scripts
    "notebooks/Chiller_PCA_Anomaly_Detection.ipynb",
    "src/data_inspection.py",
    "src/verify_modeling_scope.py",
    "src/eda_analysis.py",
    "src/preprocessing.py",
    "src/pca_detector.py",
    "src/build_notebook.py"
]

missing_count = 0
for ef in expected_files:
    if not os.path.exists(ef):
        print(f"MISSING FILE: {ef}")
        missing_count += 1
if missing_count == 0:
    print(f"\nALL {len(expected_files)} EXPECTED FILES EXIST ON DISK.")
else:
    print(f"\nWARNING: {missing_count} FILES MISSING.")

print("\nAUDIT VERIFICATION SCRIPT FINISHED SUCCESSFULLY.")
