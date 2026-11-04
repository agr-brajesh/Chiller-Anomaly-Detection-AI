import pandas as pd

val_meta = pd.read_csv("data/processed/validation_metadata.csv")
row86_meta = val_meta.iloc[86]
print("Validation row 86 metadata:")
print(row86_meta)

sl_no = row86_meta['Sl.No']
print(f"Sl.No: {sl_no}")

df_raw = pd.read_excel('data/raw/official_dataset/Analyse Chillers_AI Course.xls', sheet_name='Log_Chiller_South', skiprows=2)
raw_row = df_raw[df_raw['Sl.No'] == sl_no]
print("\nRaw row:")
for col in ['Discharge Saturation Temperature Deg C Comp 2', 'Discharge Temperature Deg C Comp 2', 'Discharge Pressure Kg /cm 2 Comp 2']:
    print(f"  {col}: {raw_row[col].values}")

