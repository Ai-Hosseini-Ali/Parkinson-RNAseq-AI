import pandas as pd


print("""
=====================================
PREPARE GSE49036 VALIDATION
8 GENE SIGNATURE
=====================================
""")


EXPR_PATH = "data/GSE49036_gene_expression.csv"
LABEL_PATH = "data/GSE49036_labels.csv"

OUTPUT = "data/GSE49036_validation_8gene.csv"


GENES = [
    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"
]


# Load expression

expr = pd.read_csv(
    EXPR_PATH,
    index_col=0
)


labels = pd.read_csv(
    LABEL_PATH
)


print("Expression:")
print(expr.shape)

print("\nLabels:")
print(labels.shape)



# Check genes

missing = [
    g for g in GENES
    if g not in expr.index
]


if missing:
    raise Exception(
        f"Missing genes: {missing}"
    )


# Select 8 genes

expr = expr.loc[GENES]


# Samples as rows

expr = expr.T

expr.index.name = "Sample"



# Merge with labels

merged = expr.merge(
    labels,
    on="Sample",
    how="inner"
)


print("\nAfter merge:")
print(merged.shape)


print("\nClasses:")
print(
    merged["Label"].value_counts()
)



# Save

merged.to_csv(
    OUTPUT,
    index=False
)


print("""
=====================================
FINISHED

Saved:
data/GSE49036_validation_8gene.csv

=====================================
""")