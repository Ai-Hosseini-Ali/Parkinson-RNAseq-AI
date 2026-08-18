import pandas as pd

# ==========================
# Load Gene Expression
# ==========================

expr = pd.read_csv(
    "data/GSE99039_gene_expression.csv",
    index_col=0
)

print("Expression:")
print(expr.shape)

# ==========================
# Load Labels
# ==========================

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)

print("\nLabels:")
print(labels.shape)

print(labels.head())

# ==========================
# Keep only CONTROL/IPD
# ==========================

labels = labels[
    labels["Disease"].isin(["CONTROL", "IPD"])
]

print("\nAfter filtering:")
print(labels["Disease"].value_counts())

# ==========================
# Keep only existing samples
# ==========================

samples = labels["Sample"].tolist()

samples = [s for s in samples if s in expr.columns]

print("\nSamples found:", len(samples))

# ==========================
# Reorder expression columns
# ==========================

expr = expr[samples]

# transpose

expr = expr.T

print("\nExpression after transpose:")
print(expr.shape)

# ==========================
# Add labels
# ==========================

expr["Disease"] = labels.set_index("Sample").loc[expr.index]["Disease"]

print("\nFinal dataset:")
print(expr.shape)

print(expr["Disease"].value_counts())

# ==========================
# Save
# ==========================

expr.to_csv(
    "data/GSE99039_final_dataset.csv"
)

print("\nSaved:")
print("data/GSE99039_final_dataset.csv")