# Makes a fake bank transactions dataset. The other files in this folder import it.
import numpy as np


def make_transactions(seed=42):
    """Return X (amount, hour) and y (0 = normal, 1 = fraud-like)."""
    rng = np.random.RandomState(seed)

    # 300 normal payments: small amounts, during the day
    normal_amount = rng.normal(loc=50, scale=15, size=300)
    normal_hour = rng.normal(loc=14, scale=3, size=300)

    # 10 fraud-like payments: very large amounts, often at night
    fraud_amount = rng.normal(loc=400, scale=60, size=10)
    fraud_hour = rng.normal(loc=3, scale=1, size=10)

    amount = np.concatenate([normal_amount, fraud_amount])
    hour = np.concatenate([normal_hour, fraud_hour])
    X = np.column_stack([amount, hour])
    X[:, 0] = np.clip(X[:, 0], 1, None)    # amounts must be positive
    X[:, 1] = np.clip(X[:, 1], 0, 23.99)   # hours must be 0 to 24
    y = np.array([0] * 300 + [1] * 10)
    return X, y


def plot_result(X, is_anomaly, title):
    """Plot normal points in blue and anomalies in red."""
    import matplotlib.pyplot as plt
    plt.scatter(X[~is_anomaly, 0], X[~is_anomaly, 1], c="blue", s=25, label="Normal")
    plt.scatter(X[is_anomaly, 0], X[is_anomaly, 1], c="red", s=50, label="Anomaly")
    plt.xlabel("Amount")
    plt.ylabel("Hour of day")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y = make_transactions()
    print("Made", len(X), "transactions:", (y == 0).sum(), "normal and", (y == 1).sum(), "fraud-like")
    print("First 3 rows (amount, hour):\n", X[:3].round(1))
