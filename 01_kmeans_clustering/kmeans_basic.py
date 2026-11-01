# Basic K-means: make fake data, group it into 3 clusters, and plot the result.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 1: create 300 sample points that form 3 natural groups
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
print("Step 1: created data with shape", X.shape)

# Step 2: run K-means with K=3
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
print("Step 2: K-means finished. Inertia (WCSS) =", round(kmeans.inertia_, 2))
print("Cluster centres:\n", kmeans.cluster_centers_)

# Step 3: plot points coloured by cluster, and mark the centres with red X
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
centres = kmeans.cluster_centers_
plt.scatter(centres[:, 0], centres[:, 1], c="red", marker="X", s=250, label="Centres")
plt.title("K-means clustering (K=3)")
plt.legend()
print("Step 3: showing plot...")
plt.show()
