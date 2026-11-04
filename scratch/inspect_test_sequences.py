import pandas as pd
import numpy as np

meta = pd.read_csv("data/processed/test_metadata.csv")
meta["Datetime"] = pd.to_datetime(meta["Timestamp"])
preds = pd.read_csv("results/metrics/phase5/test_anomaly_predictions.csv")

print("Clean Test Metadata Overview:")
print(f"Total rows: {len(meta)}")
print(meta["CHILLER TAG"].value_counts())

# Check consecutive sequences within each chiller
for chiller in meta["CHILLER TAG"].unique():
    c_df = meta[meta["CHILLER TAG"] == chiller].sort_values("Datetime")
    c_preds = preds.loc[c_df.index]
    print(f"\n--- {chiller} (N = {len(c_df)}) ---")
    c_df["time_diff"] = c_df["Datetime"].diff()
    print("Timestamps:")
    for idx, row in c_df.iterrows():
        spe = c_preds.loc[idx, "SPE"]
        flag = c_preds.loc[idx, "Is_Anomaly"]
        print(f"  Idx {idx:2d} | {row['Timestamp']} | SPE: {spe:6.4f} | Flag: {flag}")

