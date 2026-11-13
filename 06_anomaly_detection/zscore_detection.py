# Z-score detection: flag payments that are far from the average.
import numpy as np
from make_data import make_transactions, plot_result

# Step 1: get the fake transactions
X, y = make_transactions()
print("Step 1: made", len(X), "transactions")

# Step 2: Z-score = how many standard deviations a value is from the mean
z_scores = (X - X.mean(axis=0)) / X.std(axis=0)

# Step 3: flag a payment if its amount OR hour has a Z-score above 3 (in either direction)
is_anomaly = (np.abs(z_scores) > 3).any(axis=1)
print("Step 3: flagged", is_anomaly.sum(), "payments with Z-score above 3")
print("  Real fraud-like payments caught:", (is_anomaly & (y == 1)).sum(), "of", (y == 1).sum())

# Step 4: plot normal in blue, anomalies in red
print("Step 4: showing plot...")
plot_result(X, is_anomaly, "Z-score detection (|Z| > 3)")
