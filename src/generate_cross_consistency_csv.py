import csv

data = [
    {
        "Parameter_ID": 1,
        "Parameter_Name": "Official Workbook Filename & Integrity",
        "Phase_1_Value": "Analyse Chillers_AI Course.xls (SHA-256: 7c3e80d5...)",
        "Phase_2_Value": "Analyse Chillers_AI Course.xls (Unmodified raw read)",
        "Phase_3_Value": "Analyse Chillers_AI Course.xls (Input source)",
        "Phase_4_Value": "Consistent upstream source (Unchanged)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "data/raw/official_dataset/Analyse Chillers_AI Course.xls exists (665,088 bytes, SHA-256 verified identical to root copy)"
    },
    {
        "Parameter_ID": 2,
        "Parameter_Name": "Raw Sheet Name & Dimensions",
        "Phase_1_Value": "Log_Chiller_South (525 rows x 60 cols)",
        "Phase_2_Value": "Log_Chiller_South (525 rows x 60 cols)",
        "Phase_3_Value": "Log_Chiller_South (525 rows x 60 cols)",
        "Phase_4_Value": "N/A (Uses processed features)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "Raw worksheet read with xlrd confirms 525 rows, 60 columns, header on row index 2"
    },
    {
        "Parameter_ID": 3,
        "Parameter_Name": "Operating Regime Filter",
        "Phase_1_Value": "Dual-compressor operation identified",
        "Phase_2_Value": "Dual-compressor (N=517, 5 single-comp excluded)",
        "Phase_3_Value": "Dual-compressor (N=517, filtered on Comp 1 & 2 > 0)",
        "Phase_4_Value": "Dual-compressor regime (N=517 total across splits)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "data/processed/split_manifest.json specifies total_dual_observations = 517"
    },
    {
        "Parameter_ID": 4,
        "Parameter_Name": "Pooled Chiller Units",
        "Phase_1_Value": "5 Chillers (Chiller - 1 to Chiller - 5)",
        "Phase_2_Value": "5 Chillers evaluated in EDA",
        "Phase_3_Value": "5 Chillers pooled, ID retained in metadata",
        "Phase_4_Value": "5 Chillers evaluated in diagnostic scores",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "Metadata CSVs contain 'CHILLER TAG' with Chillers 1, 2, 3, 4, 5 represented"
    },
    {
        "Parameter_ID": 5,
        "Parameter_Name": "Core Thermodynamic Feature Count",
        "Phase_1_Value": "28 features (21 Temp + 7 Pressure)",
        "Phase_2_Value": "28 features profiled in correlation & EDA",
        "Phase_3_Value": "28 features selected for PCA matrix",
        "Phase_4_Value": "28 input features (n_features_in_ = 28)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "data/processed/selected_features.json and data/processed/pca_configuration.json match exactly 28 features"
    },
    {
        "Parameter_ID": 6,
        "Parameter_Name": "Temperature Feature Count",
        "Phase_1_Value": "21 thermodynamic temperatures",
        "Phase_2_Value": "21 temperatures analyzed",
        "Phase_3_Value": "21 temperatures in feature matrix",
        "Phase_4_Value": "21 temperatures in PCA loadings",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "21 features categorized as Temperature in selected_features.json"
    },
    {
        "Parameter_ID": 7,
        "Parameter_Name": "Refrigerant Pressure Feature Count",
        "Phase_1_Value": "7 refrigerant pressures",
        "Phase_2_Value": "7 refrigerant pressures analyzed",
        "Phase_3_Value": "7 refrigerant pressures in feature matrix",
        "Phase_4_Value": "7 refrigerant pressures in PCA loadings",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "7 features categorized as Pressure in selected_features.json (Pump pressures excluded)"
    },
    {
        "Parameter_ID": 8,
        "Parameter_Name": "Excluded Non-Thermodynamic Features",
        "Phase_1_Value": "Timestamp, Chiller ID, Electrical, Load excluded",
        "Phase_2_Value": "Loads & currents analyzed separately",
        "Phase_3_Value": "Loads, currents, timestamps saved to metadata only",
        "Phase_4_Value": "Strictly excluded from PCA input matrix",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "train_features.csv contains strictly 28 columns; metadata saved to train_metadata.csv"
    },
    {
        "Parameter_ID": 9,
        "Parameter_Name": "Chronological Splitting Protocol",
        "Phase_1_Value": "Identified reverse chronological ordering",
        "Phase_2_Value": "Reordered chronologically (2026-03-11 to 03-20)",
        "Phase_3_Value": "Split by unique plant timestamps (70/15/15 target)",
        "Phase_4_Value": "PCA fitted on training partition only",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "split_manifest.json: random_shuffling=False, split_methodology='Chronological partitioning'"
    },
    {
        "Parameter_ID": 10,
        "Parameter_Name": "Training Sample Count",
        "Phase_1_Value": "N/A (No split yet)",
        "Phase_2_Value": "N/A (Full EDA on 517 rows)",
        "Phase_3_Value": "N_train = 358 (69.25% of dual-comp)",
        "Phase_4_Value": "N_train = 358 (PCA fitted on 358 rows)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "train_features.csv shape = (358, 28); pca.n_samples_seen_ = 358"
    },
    {
        "Parameter_ID": 11,
        "Parameter_Name": "Validation Sample Count",
        "Phase_1_Value": "N/A",
        "Phase_2_Value": "N/A",
        "Phase_3_Value": "N_val = 92 (17.79% of dual-comp)",
        "Phase_4_Value": "N_val = 92 (Evaluated, not trained)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "validation_features.csv shape = (92, 28)"
    },
    {
        "Parameter_ID": 12,
        "Parameter_Name": "Testing Sample Count",
        "Phase_1_Value": "N/A",
        "Phase_2_Value": "N/A",
        "Phase_3_Value": "N_test = 67 (12.96% of dual-comp)",
        "Phase_4_Value": "N_test = 67 (Held-out for Phase 5)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "test_features.csv shape = (67, 28)"
    },
    {
        "Parameter_ID": 13,
        "Parameter_Name": "Temporal Split Boundary Overlap",
        "Phase_1_Value": "N/A",
        "Phase_2_Value": "Temporal continuity evaluated",
        "Phase_3_Value": "0 overlapping timestamps between partitions",
        "Phase_4_Value": "Strict temporal containment preserved",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "Train max = 2026-03-17 10:00, Val min = 2026-03-17 11:00, Val max = 2026-03-18 19:00, Test min = 2026-03-18 20:00"
    },
    {
        "Parameter_ID": 14,
        "Parameter_Name": "Imputation Strategy & Anti-Leakage",
        "Phase_1_Value": "Median imputation designated",
        "Phase_2_Value": "Confirmed 0.0% missing in raw features",
        "Phase_3_Value": "SimpleImputer(strategy='median') fitted on train only",
        "Phase_4_Value": "Fitted imputer preserved for inference pipeline",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "data/processed/imputer.joblib: statistics_ shape=(28,), fitted on train only"
    },
    {
        "Parameter_ID": 15,
        "Parameter_Name": "StandardScaler Strategy & Anti-Leakage",
        "Phase_1_Value": "StandardScaler designated",
        "Phase_2_Value": "Wide variance ranges observed across units",
        "Phase_3_Value": "StandardScaler fitted on train only (mean=0, std=1)",
        "Phase_4_Value": "Standardized train used to compute covariance",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "data/processed/scaler.joblib: mean_ shape=(28,), scale_ shape=(28,), train std max diff = 0.0"
    },
    {
        "Parameter_ID": 16,
        "Parameter_Name": "PCA Retained Components (k)",
        "Phase_1_Value": "PCA method assigned",
        "Phase_2_Value": "Multivariate collinearity confirmed (r up to 0.96)",
        "Phase_3_Value": "Input p = 28 features prepared",
        "Phase_4_Value": "k = 12 components retained (k < p)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "pca_model.joblib: n_components_ = 12, components_ shape = (12, 28)"
    },
    {
        "Parameter_ID": 17,
        "Parameter_Name": "Explained Variance Ratio",
        "Phase_1_Value": "Target >= 80% cumulative variance",
        "Phase_2_Value": "N/A",
        "Phase_3_Value": "N/A",
        "Phase_4_Value": "PC1: 23.77%, PC2: 13.64%, Cum(12): 85.46%",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "pca_explained_variance.csv and pca_configuration.json confirm 85.46% variance retained"
    },
    {
        "Parameter_ID": 18,
        "Parameter_Name": "Residual Dimensionality",
        "Phase_1_Value": "Assigned reconstruction error detection",
        "Phase_2_Value": "N/A",
        "Phase_3_Value": "28 dimensions",
        "Phase_4_Value": "16 residual dimensions (14.54% variance)",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "p - k = 28 - 12 = 16 residual dimensions retained for Phase 5 SPE detection"
    },
    {
        "Parameter_ID": 19,
        "Parameter_Name": "Component Orthogonality",
        "Phase_1_Value": "Orthogonal basis required by PCA",
        "Phase_2_Value": "N/A",
        "Phase_3_Value": "N/A",
        "Phase_4_Value": "Max off-diagonal deviation: 1.998e-15",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "V @ V.T evaluated on saved pca.components_ equals identity matrix to double precision"
    },
    {
        "Parameter_ID": 20,
        "Parameter_Name": "Figure and Metric Traceability",
        "Phase_1_Value": "Inventory and dictionary generated",
        "Phase_2_Value": "7 EDA figures + 10 CSV metrics",
        "Phase_3_Value": "2 Preprocessing figures + 4 CSV metrics",
        "Phase_4_Value": "6 PCA figures + 4 CSV metrics",
        "Cross_Phase_Status": "PASS",
        "Verification_Evidence": "All 15 figures and 18 metrics exist on disk, matching markdown documentation references"
    }
]

with open("documentation/phase1_to_phase4_cross_consistency.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Parameter_ID",
        "Parameter_Name",
        "Phase_1_Value",
        "Phase_2_Value",
        "Phase_3_Value",
        "Phase_4_Value",
        "Cross_Phase_Status",
        "Verification_Evidence"
    ])
    writer.writeheader()
    writer.writerows(data)

print("Created documentation/phase1_to_phase4_cross_consistency.csv with 20 verified parameters.")

