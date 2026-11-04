"""
Updated script to generate notebooks/Chiller_PCA_Anomaly_Detection.ipynb with Phases 1-4
"""

import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data\n",
                "\n",
                "- **Student Register Number:** 24BCE0551\n",
                "- **Track:** Multivariate and Time-Series Detection\n",
                "- **Assigned Method:** PCA-Based Multivariate Anomaly Detector\n",
                "- **Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*\n",
                "- **Official Dataset:** `data/raw/official_dataset/Analyse Chillers_AI Course.xls`\n",
                "\n",
                "---\n",
                "## Project Overview\n",
                "This notebook implements the complete research pipeline across eight structured phases. The current progress implements **Phase 1 (Dataset Understanding)**, **Phase 2 (Data Quality & EDA)**, **Phase 3 (Data Preprocessing & Anti-Data-Leakage)**, and **Phase 4 (PCA Model Development)**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup and Library Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA\n",
                "\n",
                "# Set visual styling\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "print('Libraries successfully imported.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Phase 1: Official Dataset Inspection & Discovery\n",
                "- Raw file format: Excel 97-2004 (`.xls`)\n",
                "- Primary worksheet: `Log_Chiller_South`\n",
                "- Header offset: Row index 2 (Excel row 3)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_path = os.path.join('..', 'data', 'raw', 'official_dataset', 'Analyse Chillers_AI Course.xls')\n",
                "if not os.path.exists(raw_path):\n",
                "    raw_path = os.path.join('data', 'raw', 'official_dataset', 'Analyse Chillers_AI Course.xls')\n",
                "\n",
                "df_raw = pd.read_excel(raw_path, sheet_name='Log_Chiller_South', header=2)\n",
                "df_raw.columns = [str(c).strip() for c in df_raw.columns]\n",
                "df_raw['Timestamp'] = pd.to_datetime(df_raw['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_raw['Time'].astype(str))\n",
                "print(f'Raw dataset loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns')\n",
                "print(f'Fleet distribution:\\n{df_raw[\"CHILLER TAG\"].value_counts()}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Phase 2: Data Quality Assessment & EDA Highlights\n",
                "- Completeness: 100.0% (0 missing values across all columns)\n",
                "- Duplicate rows: 0 exact duplicate rows\n",
                "- Correlation analysis confirms strong thermodynamic relationships between temperatures and pressures (e.g. Discharge Temp Comp 1 vs Discharge Pressure Comp 1, $r = +0.9578$).\n",
                "- Zero-variance features: 0 features with variance=0."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Filter dual-compressor operating regime (N = 517)\n",
                "dual_mask = (df_raw['Running Load % Comp 1'] > 0) & (df_raw['Running Load % Comp 2'] > 0)\n",
                "df_dual = df_raw[dual_mask].sort_values('Timestamp').reset_index(drop=True)\n",
                "print(f'Dual-compressor fleet size: {len(df_dual)} observations (5 single-compressor rows excluded)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Phase 3: Data Preprocessing and Anti-Data-Leakage Protocol\n",
                "\n",
                "### 4.1 Feature Identification (28 Core Thermodynamic Features)\n",
                "We restrict the PCA input space strictly to the 28 thermodynamic features directly addressing the research question."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "temp_features = [\n",
                "    'Evaporator Water In Temperature Deg C', 'Evaporator Water Out Temperature Deg C',\n",
                "    'Condenser Water In Temperature Deg C', 'Condenser Water Out Temperature Deg C',\n",
                "    'Discharge Temperature Deg C Comp 1', 'Discharge Temperature Deg C Comp 2',\n",
                "    'Suction Temperature Deg C', 'Chiler Lift Temperature Deg C Comp 1',\n",
                "    'Chiler Lift Temperature Deg C Comp 2', 'Suction Saturation Temperature Deg C',\n",
                "    'Suction Superheat deg C', 'Evaporator Approach',\n",
                "    'Discharge Saturation Temperature Deg C Comp 1', 'Discharge Saturation Temperature Deg C Comp 2',\n",
                "    'Discharge Superheat Deg C Comp 1', 'Discharge Superheat Deg C Comp 2',\n",
                "    'Condenser Approach Comp 1', 'Condenser Approach Comp 2',\n",
                "    'Liquid Temperature Deg C Comp 1', 'Liquid Temperature Deg C Comp 2',\n",
                "    'Oil Temperature Deg C'\n",
                "]\n",
                "\n",
                "press_features = [\n",
                "    'Discharge Pressure Kg /cm 2 Comp 1', 'Discharge Pressure Kg /cm 2 Comp 2',\n",
                "    'Suction Pressure Kg /cm 2', 'Chiller Lift Pressure Kg /cm 2 Comp 1',\n",
                "    'Chiller Lift Pressure Kg /cm 2 Comp 2', 'EXV Differential Pressure Kg /cm 2 Comp 1',\n",
                "    'EXV Differential Pressure Kg /cm 2 Comp 2'\n",
                "]\n",
                "\n",
                "core_28 = temp_features + press_features\n",
                "print(f'Selected {len(core_28)} core features ({len(temp_features)} Temp + {len(press_features)} Press)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.2 Chronological Partitioning (70% / 15% / 15% by Unique Timestamps)\n",
                "To prevent temporal data leakage when multiple chillers operate at the same hour, partitioning is computed on **unique timestamps**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_feat_path = os.path.join('..', 'data', 'processed', 'train_features.csv')\n",
                "if not os.path.exists(train_feat_path):\n",
                "    train_feat_path = os.path.join('data', 'processed', 'train_features.csv')\n",
                "\n",
                "val_feat_path = train_feat_path.replace('train_features.csv', 'validation_features.csv')\n",
                "test_feat_path = train_feat_path.replace('train_features.csv', 'test_features.csv')\n",
                "\n",
                "X_train = pd.read_csv(train_feat_path)\n",
                "X_val = pd.read_csv(val_feat_path)\n",
                "X_test = pd.read_csv(test_feat_path)\n",
                "\n",
                "print(f'Train features loaded: {X_train.shape} (strictly standardized on train)')\n",
                "print(f'Validation features:   {X_val.shape}')\n",
                "print(f'Test features:         {X_test.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Phase 4: PCA Model Development\n",
                "\n",
                "### 5.1 Full Spectrum PCA Exploration (Training Partition Only)\n",
                "PCA solves $\\mathbf{\\Sigma}\\mathbf{v}_i = \\lambda_i \\mathbf{v}_i$ on standardized training observations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pca_full = PCA(svd_solver='full', whiten=False)\n",
                "pca_full.fit(X_train)\n",
                "\n",
                "evr = pca_full.explained_variance_ratio_\n",
                "cev = np.cumsum(evr)\n",
                "eig_vals = pca_full.explained_variance_\n",
                "\n",
                "print('Top 5 Principal Components Explained Variance:')\n",
                "for i in range(5):\n",
                "    print(f'  PC{i+1}: Eigenvalue = {eig_vals[i]:.4f} | EVR = {evr[i]*100:.2f}% | Cumulative = {cev[i]*100:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 Scree Plot & Cumulative Explained Variance Evaluation\n",
                "Evaluate the Kaiser criterion ($\\lambda \\ge 1.0$), elbow point, and cumulative variance targets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Scree Plot\n",
                "axes[0].plot(range(1, 29), eig_vals, marker='o', color='#2b5c8f', linewidth=1.8)\n",
                "axes[0].axhline(1.0, color='#d95f02', linestyle='--', label='Kaiser Criterion (λ = 1.0)')\n",
                "axes[0].axvline(12, color='#2ca02c', linestyle=':', label='Selected Cutoff (k = 12)')\n",
                "axes[0].set_xlabel('Principal Component Index')\n",
                "axes[0].set_ylabel('Eigenvalue (Variance)')\n",
                "axes[0].set_title('PCA Scree Plot (Training Set)')\n",
                "axes[0].legend()\n",
                "\n",
                "# Cumulative Variance Plot\n",
                "axes[1].plot(range(1, 29), cev * 100, marker='s', color='#7570b3', linewidth=1.8)\n",
                "axes[1].axhline(85.0, color='#d95f02', linestyle='--', label='85% Target')\n",
                "axes[1].axvline(12, color='#2ca02c', linestyle=':', label=f'k = 12 ({cev[11]*100:.1f}%)')\n",
                "axes[1].set_xlabel('Number of Retained Components (k)')\n",
                "axes[1].set_ylabel('Cumulative Explained Variance (%)')\n",
                "axes[1].set_title('Cumulative Explained Variance Curve')\n",
                "axes[1].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.3 Final PCA Model Training (k = 12 Components, 85.46% Variance)\n",
                "We retain $k = 12$ principal components, balancing the retention of normal thermodynamic operating variance against reserving 16 residual dimensions for anomaly detection."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "k_selected = 12\n",
                "pca_model = PCA(n_components=k_selected, svd_solver='full', whiten=False)\n",
                "pca_model.fit(X_train)\n",
                "\n",
                "train_scores = pca_model.transform(X_train)\n",
                "print(f'Fitted final PCA detector with k={k_selected} components.')\n",
                "print(f'Cumulative Explained Variance: {np.sum(pca_model.explained_variance_ratio_)*100:.2f}%')\n",
                "print(f'Residual Subspace Dimensions:  {28 - k_selected} dimensions')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.4 Component Loading Analysis (Physical Meaning of PC1 and PC2)\n",
                "- **PC1 (23.77%):** High-Side / Condenser Compression Subspace (`Discharge Saturation Temp` +0.35, `Lift Pressure` +0.34, `Lift Temp` +0.30).\n",
                "- **PC2 (13.64%):** Low-Side / Evaporator Cooling Load Subspace (`Suction Pressure` +0.46, `Suction Saturation Temp` +0.38, `Superheat` -0.33)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loadings = pd.DataFrame(pca_model.components_.T, index=X_train.columns, columns=[f'PC{i+1}' for i in range(k_selected)])\n",
                "print('Top 5 Features loading into PC1 (High-Side Condenser Lift):')\n",
                "print(loadings['PC1'].abs().sort_values(ascending=False).head(5))\n",
                "\n",
                "print('\\nTop 5 Features loading into PC2 (Low-Side Evaporator Cooling Load):')\n",
                "print(loadings['PC2'].abs().sort_values(ascending=False).head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.5 Fleet Homogeneity Diagnostic (PC1 vs PC2 by Chiller Tag)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_meta_path = train_feat_path.replace('train_features.csv', 'train_metadata.csv')\n",
                "meta_train = pd.read_csv(train_meta_path)\n",
                "\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.scatterplot(x=train_scores[:, 0], y=train_scores[:, 1], hue=meta_train['CHILLER TAG'], palette='tab10', alpha=0.75, s=35)\n",
                "plt.xlabel(f'PC1 Score ({pca_model.explained_variance_ratio_[0]*100:.1f}%)')\n",
                "plt.ylabel(f'PC2 Score ({pca_model.explained_variance_ratio_[1]*100:.1f}%)')\n",
                "plt.title('Chiller Fleet Representation in Principal Subspace (PC1 vs PC2)')\n",
                "plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)\n",
                "plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)\n",
                "plt.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path = os.path.join("notebooks", "Chiller_PCA_Anomaly_Detection.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Jupyter Notebook updated successfully with Phases 1-4 at: {nb_path}")
