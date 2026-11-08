# K-means vs DBSCAN on curved (moon-shaped) data, plotted side by side.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# Step 1: make two moon shapes, then add 15 random stray points as noise
X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
rng = np.random.RandomState(0)
strays = rng.uniform(low=[-1.5, -1.0], high=[2.5, 1.5], size=(15, 2))
X = np.vstack([X, strays])
print("Step 1: created moon data with shape", X.shape)

# Step 2: K-means with K=2 (it likes round clusters, so it cuts the moons wrongly)
kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
print("Step 2: K-means done")

# Step 3: DBSCAN groups points that are close together (works for any shape)
# eps = how close points must be, min_samples = points needed to form a group
dbscan_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print("Step 3: DBSCAN found", n_clusters, "clusters and", n_noise, "noise points")

# Step 4: plot both results side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis", s=30)
axes[0].set_title("K-means (K=2): cuts the moons wrongly")

# DBSCAN marks noise points with the label -1, so we draw them in red
noise = dbscan_labels == -1
axes[1].scatter(X[~noise, 0], X[~noise, 1], c=dbscan_labels[~noise], cmap="viridis", s=30)
axes[1].scatter(X[noise, 0], X[noise, 1], c="red", marker="x", s=40, label="Noise")
axes[1].set_title("DBSCAN: follows the curves")
axes[1].legend()

print("Step 4: showing plot...")
plt.tight_layout()
plt.show()
