# Stacking: base models make predictions, then a "meta-model"
# (logistic regression here) learns how to combine those predictions.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier())),
]

print("--- Single models ---")
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name}: accuracy = {model.score(X_test, y_test):.4f}")

# cv=5 means the meta-model is trained on out-of-fold predictions,
# which avoids using predictions the base models already "saw".
stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stack.fit(X_train, y_train)
print("\n--- Ensemble ---")
print(f"stacking: accuracy = {stack.score(X_test, y_test):.4f}")
