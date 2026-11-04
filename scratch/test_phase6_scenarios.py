import os
import json
import joblib
import numpy as np
import pandas as pd

# Load artifacts
pca = joblib.load("data/processed/pca_model.joblib")
X_test = pd.read_csv("data/processed/test_features.csv")
meta_test = pd.read_csv("data/processed/test_metadata.csv")
tau = 9.2931

print(f"Loaded X_test: {X_test.shape}, threshold: {tau}")

# Clean baseline SPE
Z_clean_recon = pca.inverse_transform(pca.transform(X_test.values))
spe_clean = np.sum((X_test.values - Z_clean_recon)**2, axis=1)

# Scenario A: Spikes
# Event 1: Idx 18 (Chiller-1, 09:00), Discharge Temperature Comp 1, +4.0
# Event 2: Idx 26 (Chiller-4, 13:00), Discharge Pressure Comp 1, +4.0
# Event 3: Idx 48 (Chiller-1, 22:00), Suction Temperature, -4.0
X_spike = X_test.copy()
feat1 = "Discharge Temperature Deg C Comp 1"
feat2 = "Discharge Pressure Kg /cm 2 Comp 1"
feat3 = "Suction Temperature Deg C"

X_spike.loc[18, feat1] += 4.0
X_spike.loc[26, feat2] += 4.0
X_spike.loc[48, feat3] -= 4.0

Z_spike_recon = pca.inverse_transform(pca.transform(X_spike.values))
spe_spike = np.sum((X_spike.values - Z_spike_recon)**2, axis=1)

print("\n--- Scenario A: Spikes ---")
for idx, name in [(18, "Event 1 (Temp +4s)"), (26, "Event 2 (Press +4s)"), (48, "Event 3 (Suct -4s)")]:
    print(f"{name} at Idx {idx}: Clean SPE={spe_clean[idx]:.4f} -> Spike SPE={spe_spike[idx]:.4f} (Flagged: {spe_spike[idx] > tau})")

# Scenario B: Drift
# Chiller-1: Idx 23, 25, 27, 29, 30 (11:00 to 15:00)
# Feature: Evaporator Water In Temperature Deg C
# Ramp: 0.7, 1.4, 2.1, 2.8, 3.5
X_drift = X_test.copy()
drift_indices = [23, 25, 27, 29, 30]
feat_drift = "Evaporator Water In Temperature Deg C"
drift_offsets = [0.7, 1.4, 2.1, 2.8, 3.5]
for idx, offset in zip(drift_indices, drift_offsets):
    X_drift.loc[idx, feat_drift] += offset

Z_drift_recon = pca.inverse_transform(pca.transform(X_drift.values))
spe_drift = np.sum((X_drift.values - Z_drift_recon)**2, axis=1)

print("\n--- Scenario B: Drift ---")
for idx, offset in zip(drift_indices, drift_offsets):
    print(f"Drift Idx {idx} (+{offset:.1f}s): Clean SPE={spe_clean[idx]:.4f} -> Drift SPE={spe_drift[idx]:.4f} (Flagged: {spe_drift[idx] > tau})")

# Scenario C: Relationship Breaking
# Chiller-1: Idx 61, 63, 65 (06:00, 07:00, 08:00 on March 20)
# Feature: Discharge Pressure Kg /cm 2 Comp 1, Perturb: -3.5s
X_rel = X_test.copy()
rel_indices = [61, 63, 65]
feat_rel = "Discharge Pressure Kg /cm 2 Comp 1"
for idx in rel_indices:
    X_rel.loc[idx, feat_rel] -= 3.5

Z_rel_recon = pca.inverse_transform(pca.transform(X_rel.values))
spe_rel = np.sum((X_rel.values - Z_rel_recon)**2, axis=1)

print("\n--- Scenario C: Relationship Breaking ---")
for idx in rel_indices:
    print(f"Rel Break Idx {idx} (-3.5s Press): Clean SPE={spe_clean[idx]:.4f} -> Rel SPE={spe_rel[idx]:.4f} (Flagged: {spe_rel[idx] > tau})")

