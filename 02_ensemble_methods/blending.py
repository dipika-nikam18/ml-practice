# Blending by hand: like stacking, but the meta-model is trained on a
# separate HOLDOUT set instead of using cross-validation.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)

# Step 1: split into test set first, then split the rest into train and holdout
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_hold, y_train, y_hold = train_test_split(X_rest, y_rest, test_size=0.3, random_state=42)
print("Train:", len(X_train), "| Holdout:", len(X_hold), "| Test:", len(X_test))

base_models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
}

# Step 2: train base models on the TRAIN part only
print("\n--- Single models ---")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    print(f"{name}: accuracy = {model.score(X_test, y_test):.4f}")

# Step 3: base models predict probabilities on holdout -> features for meta-model
def blend_features(data):
    # one column per model: probability of class 1
    return np.column_stack([m.predict_proba(data)[:, 1] for m in base_models.values()])

meta_model = LogisticRegression()
meta_model.fit(blend_features(X_hold), y_hold)
print("\nMeta-model trained on holdout predictions.")

# Step 4: final prediction on the test set
print("\n--- Ensemble ---")
print(f"blending: accuracy = {meta_model.score(blend_features(X_test), y_test):.4f}")
