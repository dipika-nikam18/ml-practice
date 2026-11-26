# Apriori: find items that are often bought together (like bread + butter).
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: a small list of supermarket bills (each bill is a list of items)
bills = [
    ["bread", "butter", "milk"],
    ["bread", "butter"],
    ["bread", "butter", "eggs"],
    ["milk", "tea"],
    ["bread", "milk", "eggs"],
    ["tea", "milk"],
    ["bread", "butter", "milk", "eggs"],
    ["tea", "eggs"],
    ["bread", "butter", "tea"],
    ["milk", "eggs"],
]
items = ["bread", "butter", "milk", "eggs", "tea"]
print("Step 1: made", len(bills), "bills")

# Step 2: turn the bills into a table of True/False (was the item on the bill?)
table = pd.DataFrame([[item in bill for item in items] for bill in bills], columns=items)
print("Step 2: first 3 rows of the table:\n", table.head(3))

# Step 3: find item groups that appear in at least 30% of bills (support)
frequent = apriori(table, min_support=0.3, use_colnames=True)
print("\nStep 3: frequent item groups:\n", frequent)

# Step 4: make rules like "bread -> butter" and keep the ones with confidence >= 60%
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
rules = rules.sort_values("lift", ascending=False)

# Step 5: print the rules
# support = how often the items appear together
# confidence = when the left side is bought, how often the right side is bought too
# lift = above 1 means they are bought together more than by chance
print("\nStep 5: rules (support, confidence, lift)")
for _, row in rules.iterrows():
    left = ", ".join(sorted(row["antecedents"]))
    right = ", ".join(sorted(row["consequents"]))
    print(f"  {left} -> {right}: support={row['support']:.2f}, "
          f"confidence={row['confidence']:.2f}, lift={row['lift']:.2f}")
