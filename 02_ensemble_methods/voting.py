# Voting ensemble: several models vote on the answer.
# Hard voting = majority class. Soft voting = average of probabilities.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# Load data and split into train and test
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3 different models
models = [
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier())),
]

print("--- Single models ---")
for name, model in models:
    model.fit(X_train, y_train)
    print(f"{name}: accuracy = {model.score(X_test, y_test):.4f}")

print("\n--- Ensembles ---")
for kind in ["hard", "soft"]:
    vote = VotingClassifier(estimators=models, voting=kind)
    vote.fit(X_train, y_train)
    print(f"{kind} voting: accuracy = {vote.score(X_test, y_test):.4f}")
