import pandas as pd


print("""
==========================================
PREPARE LABELED GSE99039
8 GENE SIGNATURE
==========================================
""")


EXPR = "data/GSE99039_8gene_combat.csv"
LABEL = "data/GSE99039_labels.csv"

OUTPUT = "data/GSE99039_8gene_labeled.csv"



# load expression

expr = pd.read_csv(
    EXPR,
    index_col=0
)


labels = pd.read_csv(
    LABEL
)


print("Expression:")
print(expr.shape)


print("Labels:")
print(labels.shape)



# keep labeled samples only

labeled_samples = labels["Sample"]


expr = expr.loc[
    expr.index.intersection(
        labeled_samples
    )
]


print("\nAfter matching:")

print(expr.shape)



expr.to_csv(
    OUTPUT
)


print("""
==========================================
FINISHED

Saved:
data/GSE99039_8gene_labeled.csv

==========================================
""")