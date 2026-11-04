"""
Verification of Modeling Scope and Thermodynamic Features
=========================================================
Checks:
1. 517 dual-compressor observations.
2. 28 core thermodynamic features.
3. Measurement range comparison across Chillers 1-5.
4. Timestamp distribution, concurrency, and chronological partitioning implications.
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join("data", "raw", "official_dataset", "Analyse Chillers_AI Course.xls")

def verify():
    df = pd.read_excel(DATA_PATH, sheet_name="Log_Chiller_South", header=2)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 28 Thermodynamic features
    temp_features = [
        "Evaporator Water In Temperature Deg C",
        "Evaporator Water Out Temperature Deg C",
        "Condenser Water In Temperature Deg C",
        "Condenser Water Out Temperature Deg C",
        "Discharge Temperature Deg C Comp 1",
        "Discharge Temperature Deg C Comp 2",
        "Suction Temperature Deg C",
        "Chiler Lift Temperature Deg C Comp 1",
        "Chiler Lift Temperature Deg C Comp 2",
        "Suction Saturation Temperature Deg C",
        "Suction Superheat deg C",
        "Evaporator Approach",
        "Discharge Saturation Temperature Deg C Comp 1",
        "Discharge Saturation Temperature Deg C Comp 2",
        "Discharge Superheat Deg C Comp 1",
        "Discharge Superheat Deg C Comp 2",
        "Condenser Approach Comp 1",
        "Condenser Approach Comp 2",
        "Liquid Temperature Deg C Comp 1",
        "Liquid Temperature Deg C Comp 2",
        "Oil Temperature Deg C"
    ]

    press_features = [
        "Discharge Pressure Kg /cm 2 Comp 1",
        "Discharge Pressure Kg /cm 2 Comp 2",
        "Suction Pressure Kg /cm 2",
        "Chiller Lift Pressure Kg /cm 2 Comp 1",
        "Chiller Lift Pressure Kg /cm 2 Comp 2",
        "EXV Differential Pressure Kg /cm 2 Comp 1",
        "EXV Differential Pressure Kg /cm 2 Comp 2"
    ]

    all_28 = temp_features + press_features

    print("=" * 80)
    print("1. VERIFICATION OF 28 THERMODYNAMIC FEATURES")
    print("=" * 80)
    print(f"Temperature variables count: {len(temp_features)}")
    print(f"Pressure variables count:    {len(press_features)}")
    print(f"Total core features count:   {len(all_28)}")

    missing_in_df = [f for f in all_28 if f not in df.columns]
    print(f"Missing from dataframe: {missing_in_df} (Must be empty)")
    assert len(missing_in_df) == 0, "Some features missing!"

    # Operating regime filtering
    print("\n" + "=" * 80)
    print("2. DUAL-COMPRESSOR FILTERING (N = 517)")
    print("=" * 80)
    dual_mask = (df["Running Load % Comp 1"] > 0) & (df["Running Load % Comp 2"] > 0)
    print(f"Total raw rows: {len(df)}")
    print(f"Dual-compressor active rows: {dual_mask.sum()}")
    print(f"Single-compressor rows: {(~dual_mask).sum()}")
    
    df_dual = df[dual_mask].copy()
    assert len(df_dual) == 517, f"Expected 517 rows, got {len(df_dual)}"
    print("-> Dual-compressor count of 517 EXACTLY confirmed.")

    # Breakdown per chiller
    print("\nDual-compressor counts per chiller:")
    print(df_dual["CHILLER TAG"].value_counts())

    # Check missing values in the 28 columns for these 517 rows
    null_counts = df_dual[all_28].isna().sum().sum()
    print(f"\nMissing values across all 28 variables in the 517 rows: {null_counts}")
    assert null_counts == 0, "Missing values found!"

    # 3. Measurement range comparison across chillers
    print("\n" + "=" * 80)
    print("3. COMPARISON OF DISTRIBUTIONS ACROSS CHILLERS (Key Features)")
    print("=" * 80)
    key_vars = [
        "Evaporator Water Out Temperature Deg C",
        "Condenser Water In Temperature Deg C",
        "Suction Pressure Kg /cm 2",
        "Discharge Pressure Kg /cm 2 Comp 1",
        "Suction Temperature Deg C",
        "Oil Temperature Deg C"
    ]

    for kv in key_vars:
        print(f"\nFeature: '{kv}'")
        summary_per_chiller = df_dual.groupby("CHILLER TAG")[kv].agg(["count", "mean", "std", "min", "median", "max"])
        print(summary_per_chiller.to_string())

    # 4. Timestamp analysis and chronological concurrency
    print("\n" + "=" * 80)
    print("4. TIMESTAMP STRUCTURE & CHRONOLOGICAL PARTITIONING")
    print("=" * 80)
    df_dual["Timestamp"] = pd.to_datetime(
        df_dual["Date"].dt.strftime("%Y-%m-%d") + " " + df_dual["Time"].astype(str)
    )
    df_sorted = df_dual.sort_values("Timestamp").reset_index(drop=True)
    
    unique_ts = df_sorted["Timestamp"].nunique()
    print(f"Total rows in pooled dataset: {len(df_sorted)}")
    print(f"Total unique timestamps:      {unique_ts}")
    print(f"Earliest timestamp:           {df_sorted['Timestamp'].min()}")
    print(f"Latest timestamp:             {df_sorted['Timestamp'].max()}")

    # Distribution of chillers active per timestamp
    ts_group = df_sorted.groupby("Timestamp")["CHILLER TAG"].count()
    print("\nNumber of concurrent chiller records per timestamp:")
    print(ts_group.value_counts().sort_index())

    # Check for same timestamp across train/val/test boundary
    # If we split chronologically e.g. 70% / 15% / 15%:
    split_train = int(unique_ts * 0.70)
    split_val = int(unique_ts * 0.85)
    sorted_unique_ts = np.sort(df_sorted["Timestamp"].unique())
    
    train_end_ts = sorted_unique_ts[split_train - 1]
    val_start_ts = sorted_unique_ts[split_train]
    val_end_ts = sorted_unique_ts[split_val - 1]
    test_start_ts = sorted_unique_ts[split_val]

    print("\nChronological Split Simulation (Based on Unique Timestamps):")
    print(f"Train timestamp interval: {sorted_unique_ts[0]} to {train_end_ts} ({split_train} hours)")
    print(f"Val timestamp interval:   {val_start_ts} to {val_end_ts} ({split_val - split_train} hours)")
    print(f"Test timestamp interval:  {test_start_ts} to {sorted_unique_ts[-1]} ({len(sorted_unique_ts) - split_val} hours)")

    train_rows = (df_sorted["Timestamp"] <= train_end_ts).sum()
    val_rows = ((df_sorted["Timestamp"] >= val_start_ts) & (df_sorted["Timestamp"] <= val_end_ts)).sum()
    test_rows = (df_sorted["Timestamp"] >= test_start_ts).sum()
    print(f"Observations partitioned: Train={train_rows} ({100*train_rows/517:.1f}%), Val={val_rows} ({100*val_rows/517:.1f}%), Test={test_rows} ({100*test_rows/517:.1f}%)")
    print(f"Total partitioned: {train_rows + val_rows + test_rows} / 517")

if __name__ == "__main__":
    verify()
