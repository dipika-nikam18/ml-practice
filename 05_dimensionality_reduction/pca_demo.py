# PCA: squeeze the digits dataset from 64 features down to 2 and plot it.
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Step 1: load the digits dataset (1797 small 8x8 images = 64 numbers each)
digits = load_digits()
X, y = digits.data, digits.target
print("Step 1: loaded digits with shape", X.shape)

# Step 2: reduce 64 features to 2 with PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print("Step 2: new shape is", X_2d.shape)

# Step 3: explained variance = how much of the original information each part keeps
print("Step 3: explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("  Total kept by 2 parts:", round(pca.explained_variance_ratio_.sum() * 100, 1), "%")

# Step 4: plot the 2D points, coloured by the real digit
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("Part 1 (PC1)")
plt.ylabel("Part 2 (PC2)")
plt.title("PCA of digits: 64 features -> 2")
print("Step 4: showing plot...")
plt.show()
