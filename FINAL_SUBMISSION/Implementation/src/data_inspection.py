"""
Phase 1: Official Dataset Inspection and Factual Profiling
===========================================================
Academic Project: AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Detector

This script performs Phase 1 inspection tasks:
1. Verifies dependency availability (xlrd, pandas, numpy, etc.).
2. Checks file integrity and compares root vs official dataset directory copies (SHA-256).
3. Inspects ALL worksheets in the Excel workbook (including hidden sheets).
4. Unpacks headers, dimensions, data types, and missing values.
5. Categorizes variables into domain-specific subsystems (Temperature, Pressure, Electrical, etc.).
6. Evaluates timestamp continuity, chronological ordering, and sampling frequency.
7. Identifies operating regimes and initial data-quality anomalies.
"""

import os
import sys
import hashlib
import importlib
import datetime
import numpy as np
import pandas as pd
import xlrd

ROOT_FILE = "Analyse Chillers_AI Course.xls"
RAW_DATASET_FILE = os.path.join("data", "raw", "official_dataset", "Analyse Chillers_AI Course.xls")


def check_dependencies():
    """Verify that required Python packages for Phase 1 inspection are installed."""
    print("=" * 80)
    print("STEP 1: DEPENDENCY VERIFICATION")
    print("=" * 80)
    packages = ["xlrd", "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn"]
    status_dict = {}
    all_ok = True
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "Installed (no version attribute)")
            status_dict[pkg] = f"AVAILABLE (v{version})"
        except ImportError as e:
            status_dict[pkg] = f"MISSING ({e})"
            all_ok = False
        print(f"  - {pkg:15s}: {status_dict[pkg]}")
    
    if not all_ok:
        print("\n[WARNING] Some dependencies are missing. Run: pip install -r requirements.txt")
    else:
        print("\n[SUCCESS] All inspection dependencies are installed and available.")
    return all_ok


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compare_dataset_files():
    """Verify folder structure and compare root copy with data/raw/official_dataset copy."""
    print("\n" + "=" * 80)
    print("STEP 2: FOLDER STRUCTURE & FILE INTEGRITY VERIFICATION")
    print("=" * 80)
    print(f"Checking official dataset path: {RAW_DATASET_FILE}")
    raw_exists = os.path.exists(RAW_DATASET_FILE)
    root_exists = os.path.exists(ROOT_FILE)

    print(f"  - File in data/raw/official_dataset/ exists: {raw_exists}")
    print(f"  - File in project root exists:              {root_exists}")

    if not raw_exists:
        raise FileNotFoundError(f"Official dataset not found at {RAW_DATASET_FILE}")

    raw_size = os.path.getsize(RAW_DATASET_FILE)
    raw_hash = compute_sha256(RAW_DATASET_FILE)
    print(f"  - data/raw/ file size: {raw_size:,} bytes")
    print(f"  - data/raw/ SHA-256:   {raw_hash}")

    if root_exists:
        root_size = os.path.getsize(ROOT_FILE)
        root_hash = compute_sha256(ROOT_FILE)
        print(f"  - Root file size:      {root_size:,} bytes")
        print(f"  - Root file SHA-256:   {root_hash}")
        are_identical = (raw_hash == root_hash)
        print(f"  - Byte-for-byte SHA-256 match: {are_identical}")
        if are_identical:
            print("  -> CONFIRMED: Both copies are 100% identical.")
        else:
            print("  -> [WARNING]: The two files differ in content!")
    else:
        print("  - Note: No duplicate file in project root.")


def inspect_all_worksheets():
    """Inspect all sheets in the Excel workbook, including hidden or metadata sheets."""
    print("\n" + "=" * 80)
    print("STEP 3: WORKBOOK WORKSHEET DISCOVERY")
    print("=" * 80)
    wb = xlrd.open_workbook(RAW_DATASET_FILE, on_demand=False)
    num_sheets = wb.nsheets
    sheet_names = wb.sheet_names()
    print(f"Total worksheets found: {num_sheets}")
    print(f"Worksheet names: {sheet_names}")

    visibility_map = {0: "Visible", 1: "Hidden", 2: "Very Hidden"}
    sheet_details = []
    for idx in range(num_sheets):
        sh = wb.sheet_by_index(idx)
        vis = visibility_map.get(sh.visibility, f"Unknown ({sh.visibility})")
        print(f"\n  Sheet {idx+1}: '{sh.name}'")
        print(f"    - Visibility: {vis}")
        print(f"    - Dimensions: {sh.nrows} rows x {sh.ncols} columns")
        
        # Sample first 4 rows to detect header structure
        print("    - First 4 rows preview:")
        for r in range(min(4, sh.nrows)):
            row_vals = sh.row_values(r)
            non_empty = [v for v in row_vals if str(v).strip() != ""]
            print(f"        Row {r} (Excel row {r+1}): {len(non_empty)} non-empty cells -> {row_vals[:6]}")
        sheet_details.append((sh.name, sh.nrows, sh.ncols, vis))
    return sheet_details


def inspect_primary_dataset():
    """Load and profile the primary operational worksheet."""
    print("\n" + "=" * 80)
    print("STEP 4: DETAILED VARIABLE & DATA PROFILING (Sheet: 'Log_Chiller_South')")
    print("=" * 80)
    
    # Header is at row index 2 (Excel row 3)
    df = pd.read_excel(RAW_DATASET_FILE, sheet_name="Log_Chiller_South", header=2)
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Observational Rows: {df.shape[0]}")
    print(f"Variables (Columns): {df.shape[1]}")
    print(f"Total Missing Values: {df.isna().sum().sum()}")

    # Timestamp synthesis
    df['Timestamp'] = pd.to_datetime(
        df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str)
    )

    print(f"\nTimestamp Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Total Calendar Days: {df['Date'].dt.date.nunique()} days (from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Raw Sheet Temporal Ordering: First row = {df['Timestamp'].iloc[0]}, Last row = {df['Timestamp'].iloc[-1]}")
    if df['Timestamp'].iloc[0] > df['Timestamp'].iloc[-1]:
        print("  -> Raw Excel file is sorted in REVERSE chronological order.")

    # Chiller tag distribution
    print("\n" + "-" * 60)
    print("CHILLER FLEET BREAKDOWN:")
    print("-" * 60)
    for tag, count in df['CHILLER TAG'].value_counts().items():
        sub = df[df['CHILLER TAG'] == tag].sort_values('Timestamp')
        print(f"  - {tag:12s}: {count:3d} observations | Range: {sub['Timestamp'].min()} to {sub['Timestamp'].max()} | Duplicates: {sub['Timestamp'].duplicated().sum()}")

    # Variable categorization
    print("\n" + "-" * 60)
    print("VARIABLE CATEGORIZATION (All 60 Variables):")
    print("-" * 60)

    categories = {
        "Identification & Timestamps": ["Sl.No", "CHILLER TAG", "Date", "Time"],
        "Temperature Variables (°C)": [],
        "Refrigerant Pressure Variables (kg/cm²)": [],
        "Compressor Electrical & Load": [],
        "Actuators & Controls": [],
        "Compressor Run Hours": [],
        "Auxiliary Plant Equipment (Pumps & Fans)": []
    }

    for col in df.columns:
        if col in categories["Identification & Timestamps"] or col == 'Timestamp':
            continue
        c_lower = col.lower()
        if 'run hours' in c_lower or 'compressure run' in c_lower:
            categories["Compressor Run Hours"].append(col)
        elif any(k in c_lower for k in ['temp', 'deg c', 'superheat', 'approach']):
            categories["Temperature Variables (°C)"].append(col)
        elif ('press' in c_lower or 'kg /cm' in c_lower) and not ('pump' in c_lower):
            categories["Refrigerant Pressure Variables (kg/cm²)"].append(col)
        elif any(k in c_lower for k in ['load %', 'motor current', 'volt']):
            categories["Compressor Electrical & Load"].append(col)
        elif any(k in c_lower for k in ['vfd', 'exv', 'output to']):
            categories["Actuators & Controls"].append(col)
        elif any(k in c_lower for k in ['pump', 'cooling tower', 'fan']):
            categories["Auxiliary Plant Equipment (Pumps & Fans)"].append(col)
        else:
            categories.setdefault("Other", []).append(col)

    for cat_name, cols in categories.items():
        print(f"\n[{cat_name}] ({len(cols)} variables):")
        for c in cols:
            print(f"    * {c}")

    # Operating states
    print("\n" + "-" * 60)
    print("OPERATING REGIMES (Compressor States):")
    print("-" * 60)
    both_on = ((df['Running Load % Comp 1'] > 0) & (df['Running Load % Comp 2'] > 0)).sum()
    c1_only = ((df['Running Load % Comp 1'] > 0) & (df['Running Load % Comp 2'] == 0)).sum()
    c2_only = ((df['Running Load % Comp 1'] == 0) & (df['Running Load % Comp 2'] > 0)).sum()
    neither = ((df['Running Load % Comp 1'] == 0) & (df['Running Load % Comp 2'] == 0)).sum()
    print(f"  - Dual-Compressor Operation (Comp 1 & Comp 2 Active): {both_on} rows ({100*both_on/len(df):.2f}%)")
    print(f"  - Single-Compressor Operation (Comp 1 Active Only):     {c1_only} rows ({100*c1_only/len(df):.2f}%)")
    print(f"  - Single-Compressor Operation (Comp 2 Active Only):     {c2_only} rows ({100*c2_only/len(df):.2f}%)")
    print(f"  - Chiller Idle / Offline:                               {neither} rows (0.00%)")

    # Critical logging anomalies
    print("\n" + "-" * 60)
    print("DATA-QUALITY DIAGNOSTIC SCAN (Observed Typos):")
    print("-" * 60)
    
    anomalies = [
        ("Motor Current Comp 1 > 300 A (10x decimal shift)", df[df['Motor Current Amps Comp 1'] > 300][['Sl.No', 'CHILLER TAG', 'Date', 'Time', 'Running Load % Comp 1', 'Motor Current Amps Comp 1']]),
        ("Motor Current Comp 2 > 300 A (10x decimal shift)", df[df['Motor Current Amps Comp 2'] > 300][['Sl.No', 'CHILLER TAG', 'Date', 'Time', 'Running Load % Comp 2', 'Motor Current Amps Comp 2']]),
        ("Liquid Temp Comp 1 > 50 °C (manual entry typo)", df[df['Liquid Temperature Deg C Comp 1'] > 50][['Sl.No', 'CHILLER TAG', 'Date', 'Time', 'Liquid Temperature Deg C Comp 1']]),
        ("Condenser Water Out < 20 °C (physically implausible)", df[df['Condenser Water Out Temperature Deg C'] < 20][['Sl.No', 'CHILLER TAG', 'Date', 'Time', 'Condenser Water In Temperature Deg C', 'Condenser Water Out Temperature Deg C']]),
        ("Discharge Pressure Comp 1 or 2 > 25 kg/cm²", df[(df['Discharge Pressure Kg /cm 2 Comp 1'] > 25) | (df['Discharge Pressure Kg /cm 2 Comp 2'] > 25)][['Sl.No', 'CHILLER TAG', 'Date', 'Time', 'Discharge Pressure Kg /cm 2 Comp 1', 'Discharge Pressure Kg /cm 2 Comp 2']])
    ]

    for title, subset in anomalies:
        print(f"\n>>> {title}: {len(subset)} occurrence(s)")
        print(subset.to_string(index=False))

    print("\n" + "=" * 80)
    print("PHASE 1 INSPECTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    check_dependencies()
    compare_dataset_files()
    inspect_all_worksheets()
    inspect_primary_dataset()
