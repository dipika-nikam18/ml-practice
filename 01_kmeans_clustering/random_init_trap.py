# The random-init trap: with init="random" and n_init=1, a bad starting
# position can give a bad result. k-means++ picks smarter starting centres.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=7)
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Run both methods with many seeds and print the WCSS (lower is better)
random_results, pp_results = [], []
print("Seed | random WCSS | k-means++ WCSS")
for s in seeds:
    r = KMeans(n_clusters=4, init="random", n_init=1, random_state=s).fit(X)
    p = KMeans(n_clusters=4, init="k-means++", n_init=1, random_state=s).fit(X)
    random_results.append(r)
    pp_results.append(p)
    print(f"{s:4d} | {r.inertia_:11.2f} | {p.inertia_:.2f}")

# Pick the WORST random run to show the trap
worst = max(random_results, key=lambda m: m.inertia_)
best_pp = min(pp_results, key=lambda m: m.inertia_)
print("\nWorst random run WCSS:", round(worst.inertia_, 2))
print("Best k-means++ run WCSS:", round(best_pp.inertia_, 2))

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, model, title in [(axes[0], worst, "init='random' (worst seed)"),
                         (axes[1], best_pp, "init='k-means++'")]:
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=25)
    c = model.cluster_centers_
    ax.scatter(c[:, 0], c[:, 1], c="red", marker="X", s=200)
    ax.set_title(f"{title}\nWCSS = {model.inertia_:.1f}")
plt.tight_layout()
plt.show()
