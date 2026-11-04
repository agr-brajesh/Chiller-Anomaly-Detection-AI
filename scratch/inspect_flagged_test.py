import pandas as pd
import numpy as np

df_test_preds = pd.read_csv("results/metrics/phase5/test_anomaly_predictions.csv")
df_test_feat = pd.read_csv("data/processed/test_features.csv")
df_contrib = pd.read_csv("results/metrics/phase5/feature_residual_contributions.csv")

flagged = df_test_preds[df_test_preds["Is_Anomaly"] == 1]
print("Flagged Test Observations:")
for idx, row in flagged.iterrows():
    print("="*60)
    print(f"Index: {idx} | Chiller: {row['CHILLER TAG']} | Timestamp: {row['Timestamp']} | SPE: {row['SPE']:.4f}")
    c_row = df_contrib.iloc[idx].drop(["CHILLER TAG", "Timestamp", "Total_SPE"])
    c_row_pct = (c_row / row['SPE']) * 100.0
    top_c = c_row_pct.sort_values(ascending=False).head(5)
    print("Top 5 Contributing Sensors (% of SPE):")
    for feat, pct in top_c.items():
        z_val = df_test_feat.iloc[idx][feat]
        print(f"  {feat:45s}: {pct:5.2f}% (Standardized z = {z_val:+.2f})")

