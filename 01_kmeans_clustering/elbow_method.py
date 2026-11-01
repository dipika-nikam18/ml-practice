# Elbow method: try K = 1..10 and look for the "bend" in the WCSS curve.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# WCSS = within-cluster sum of squares (sklearn calls it inertia).
# Smaller means points are closer to their centre.
k_values = list(range(1, 11))
wcss = []
for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    wcss.append(model.inertia_)
    print(f"K={k:2d}  WCSS={model.inertia_:.2f}")

# Find the elbow: the K where the curve suddenly flattens.
# We compare the drop before K with the drop after K.
# A big ratio means big improvement up to K, and almost none after.
ratio = {}
for k in range(2, 10):
    drop_before = wcss[k - 2] - wcss[k - 1]   # WCSS(K-1) - WCSS(K)
    drop_after = wcss[k - 1] - wcss[k]        # WCSS(K) - WCSS(K+1)
    ratio[k] = drop_before / drop_after
elbow_k = max(ratio, key=ratio.get)
print("\nThe elbow is at K =", elbow_k, "(the data really has 4 groups)")

plt.plot(k_values, wcss, marker="o")
plt.scatter([elbow_k], [wcss[elbow_k - 1]], c="red", s=200, zorder=3, label=f"Elbow (K={elbow_k})")
plt.xlabel("Number of clusters K")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow method")
plt.legend()
plt.show()
