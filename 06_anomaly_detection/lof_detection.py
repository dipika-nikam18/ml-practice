# Local Outlier Factor (LOF): a point is odd if its area is much emptier than its neighbours' areas.
from sklearn.neighbors import LocalOutlierFactor
from make_data import make_transactions, plot_result

# Step 1: get the fake transactions
X, y = make_transactions()
print("Step 1: made", len(X), "transactions")

# Step 2: fit the model. n_neighbors = how many close points we compare with
model = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
labels = model.fit_predict(X)  # -1 means anomaly, 1 means normal
is_anomaly = labels == -1
print("Step 2: LOF flagged", is_anomaly.sum(), "anomalies")
print("  Real fraud-like payments caught:", (is_anomaly & (y == 1)).sum(), "of", (y == 1).sum())

# Step 3: plot normal in blue, anomalies in red
print("Step 3: showing plot...")
plot_result(X, is_anomaly, "Local Outlier Factor (LOF)")
