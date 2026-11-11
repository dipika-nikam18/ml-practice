# Compare Z-score, Isolation Forest and LOF: how many anomalies does each one find?
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from make_data import make_transactions

# Step 1: get the fake transactions (10 of them are really fraud-like)
X, y = make_transactions()
print("Step 1: made", len(X), "transactions,", (y == 1).sum(), "are really fraud-like\n")

# Step 2: run each method and keep a True/False list of flagged payments
z_scores = (X - X.mean(axis=0)) / X.std(axis=0)
found = {
    "Z-score": (np.abs(z_scores) > 3).any(axis=1),
    "Isolation Forest": IsolationForest(contamination=0.03, random_state=42).fit_predict(X) == -1,
    "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.03).fit_predict(X) == -1,
}

# Step 3: print how many each method found, and how many were real fraud
print("Step 3: results")
print(f"{'Method':<18}{'Flagged':>8}{'Real fraud caught':>20}")
for name, flags in found.items():
    print(f"{name:<18}{flags.sum():>8}{(flags & (y == 1)).sum():>20}")
