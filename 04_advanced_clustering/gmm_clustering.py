# Gaussian Mixture Model (GMM): soft clustering that gives a probability for each group.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Step 1: create 300 points in 3 groups
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.2, random_state=42)
print("Step 1: created data with shape", X.shape)

# Step 2: fit a GMM with 3 components (groups)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
print("Step 2: GMM finished")

# Step 3: print the probability of each group for the first 5 points
# Each row adds up to 1. A high number means the point likely belongs to that group.
probs = gmm.predict_proba(X)
print("Step 3: probability of each group for the first 5 points")
for i in range(5):
    print(f"  Point {i}: group 0 = {probs[i, 0]:.3f}, group 1 = {probs[i, 1]:.3f}, group 2 = {probs[i, 2]:.3f}")

# Step 4: plot points coloured by group. Bigger points = less sure about the group.
confidence = np.max(probs, axis=1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20 + 80 * (1 - confidence))
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c="red", marker="X", s=250, label="Group centres")
plt.title("GMM clustering (big points = less sure)")
plt.legend()
print("Step 4: showing plot...")
plt.show()
