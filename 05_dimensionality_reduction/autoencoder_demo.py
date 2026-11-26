# Autoencoder: a small neural network that squeezes digits to 2 numbers and rebuilds them.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

torch.manual_seed(42)

# Step 1: load digits and scale pixel values from 0..16 to 0..1
digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = digits.target
print("Step 1: loaded digits with shape", tuple(X.shape))

# Step 2: build the network
# Encoder: 64 -> 32 -> 2 (squeeze). Decoder: 2 -> 32 -> 64 (rebuild).
encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
decoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
model = nn.Sequential(encoder, decoder)
loss_fn = nn.MSELoss()  # how different is the rebuilt image from the original
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("Step 2: built the autoencoder")

# Step 3: train. Each loop: rebuild images, measure the error, improve the weights.
print("Step 3: training...")
for epoch in range(1, 301):
    rebuilt = model(X)
    loss = loss_fn(rebuilt, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: loss = {loss.item():.4f}")

# Step 4: get the 2 numbers per image and the rebuilt images
with torch.no_grad():
    codes = encoder(X).numpy()
    rebuilt = model(X).numpy()

# Step 5: show 8 original images (top row) and their rebuilt versions (bottom row)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for i in range(8):
    axes[0, i].imshow(X[i].numpy().reshape(8, 8), cmap="gray")
    axes[1, i].imshow(rebuilt[i].reshape(8, 8), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_title("Original", loc="left")
axes[1, 0].set_title("Rebuilt", loc="left")
plt.tight_layout()
print("Step 5: showing original vs rebuilt images (close it to see the 2D map)...")
plt.show()

# Step 6: plot the 2 numbers (the "code") for every digit
scatter = plt.scatter(codes[:, 0], codes[:, 1], c=y, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.title("Autoencoder: digits squeezed to 2 numbers")
print("Step 6: showing 2D map...")
plt.show()
