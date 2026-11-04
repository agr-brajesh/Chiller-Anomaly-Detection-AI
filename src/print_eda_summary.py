import pandas as pd

df_pair = pd.read_csv('results/metrics/eda/temperature_pressure_pairwise_correlations.csv')
print("Top 10 strongest positive correlations:")
for idx, r in df_pair.head(10).iterrows():
    print(f"  {idx+1:2d}. {r['Temperature_Variable'][:35]} vs {r['Pressure_Variable'][:30]} | Pearson r = {r['Pearson_r']:+.4f} | Spearman rho = {r['Spearman_rho']:+.4f}")

print("\nTop 5 strongest negative correlations:")
for idx, r in df_pair.tail(5).iterrows():
    print(f"  {idx+1:2d}. {r['Temperature_Variable'][:35]} vs {r['Pressure_Variable'][:30]} | Pearson r = {r['Pearson_r']:+.4f} | Spearman rho = {r['Spearman_rho']:+.4f}")

print("\nSensor Quality & Outliers Summary (Top 8 Outlier-Prone Features):")
df_qual = pd.read_csv('results/metrics/eda/sensor_quality_summary.csv')
df_top_outliers = df_qual.sort_values('IQR_Outlier_Count', ascending=False).head(8)
for idx, r in df_top_outliers.iterrows():
    print(f"  * {r['Feature_Name'][:40]:40s} | Variance = {r['Variance']:10.4f} | IQR Outliers = {r['IQR_Outlier_Count']:3d} ({r['IQR_Outlier_Pct']:4.1f}%) | Min={r['Min_Value']} Max={r['Max_Value']}")

