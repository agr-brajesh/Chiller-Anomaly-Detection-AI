import pandas as pd
import numpy as np

df_val = pd.read_csv("results/metrics/phase5/validation_reconstruction_errors.csv")
top_val = df_val.sort_values(by="SPE", ascending=False).head(5)
print("Top 5 Validation Observations with Highest SPE:")
print(top_val[["CHILLER TAG", "Timestamp", "SPE", "MSE_residual"]])

# Let's check raw features of that top observation
df_val_raw = pd.read_csv("data/processed/validation_features.csv")
top_idx = top_val.index[0]
print(f"\nTop index {top_idx} details:")
row = df_val_raw.iloc[top_idx]
print(row[row.abs() > 3.0]) # standardized features with z-score > 3

