# Isolation Forest: anomalies are easy to "isolate", so they get separated quickly.
from sklearn.ensemble import IsolationForest
from make_data import make_transactions, plot_result

# Step 1: get the fake transactions
X, y = make_transactions()
print("Step 1: made", len(X), "transactions")

# Step 2: fit the model. contamination = the share of data we expect to be anomalies
model = IsolationForest(contamination=0.03, random_state=42)
labels = model.fit_predict(X)  # -1 means anomaly, 1 means normal
is_anomaly = labels == -1
print("Step 2: Isolation Forest flagged", is_anomaly.sum(), "anomalies")
print("  Real fraud-like payments caught:", (is_anomaly & (y == 1)).sum(), "of", (y == 1).sum())

# Step 3: plot normal in blue, anomalies in red
print("Step 3: showing plot...")
plot_result(X, is_anomaly, "Isolation Forest")
