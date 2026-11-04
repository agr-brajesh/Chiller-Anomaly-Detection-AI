import joblib
import numpy as np
import pandas as pd

pca = joblib.load("data/processed/pca_model.joblib")
X_test = pd.read_csv("data/processed/test_features.csv")
tau = 9.2931

drift_indices = [23, 25, 27, 29, 30]
feat_drift = "Evaporator Water Out Temperature Deg C"
drift_offsets = [0.7, 1.4, 2.1, 2.8, 3.5]

X_d = X_test.copy()
for idx, off in zip(drift_indices, drift_offsets):
    X_d.loc[idx, feat_drift] += off

recon = pca.inverse_transform(pca.transform(X_d.values))
spe_d = np.sum((X_d.values - recon)**2, axis=1)

print("Drift ramp on Evaporator Water Out Temperature Deg C:")
for idx, off in zip(drift_indices, drift_offsets):
    spe = spe_d[idx]
    print(f"  Idx {idx:2d} (+{off:.1f}s): SPE = {spe:6.4f} | Flagged: {spe > tau}")

