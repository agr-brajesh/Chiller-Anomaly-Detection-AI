import joblib
import numpy as np
import pandas as pd

pca = joblib.load("data/processed/pca_model.joblib")
X_test = pd.read_csv("data/processed/test_features.csv")
tau = 9.2931

print("Testing which features cause high SPE under drift:")
for feat in X_test.columns:
    X_d = X_test.copy()
    # Apply +3.5 to index 30
    X_d.loc[30, feat] += 3.5
    recon = pca.inverse_transform(pca.transform(X_d.values))
    spe = np.sum((X_d.values - recon)**2, axis=1)[30]
    # Check loading in top 12 PCs vs residual
    # Residual projection matrix Q = I - V V^T. The diagonal element Q_jj indicates how much of feature j variance goes into residuals!
    V = pca.components_ # (12, 28)
    # The leverage / communality of feature j in top 12 PCs is h_j = sum_{k=1}^12 V_{kj}^2
    # The residual fraction is 1 - h_j
    feat_idx = list(X_test.columns).index(feat)
    h_j = np.sum(V[:, feat_idx]**2)
    res_fraction = 1.0 - h_j
    print(f"  {feat:42s} | Res Fraction: {res_fraction:5.3f} | SPE: {spe:6.3f} | Flagged: {spe > tau}")

