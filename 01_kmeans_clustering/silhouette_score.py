# Silhouette score: from -1 to 1. Higher means clusters are better separated.
# It needs at least 2 clusters, so we try K = 2..10.
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

scores = {}
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    print(f"K={k:2d}  silhouette score = {scores[k]:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nBest K = {best_k} (score {scores[best_k]:.4f})")
