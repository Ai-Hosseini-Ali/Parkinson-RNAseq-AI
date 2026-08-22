import pandas as pd
import os


print("""
==========================================
PREPARE GSE99039
8 GENE SIGNATURE
FOR COMBAT
==========================================
""")


INPUT = "data/GSE99039_gene_expression.csv"

OUTPUT = "data/GSE99039_8gene_combat.csv"


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


# ==========================
# Load
# ==========================

print("Loading expression...")

df = pd.read_csv(
    INPUT,
    index_col=0
)


print("Original:")
print(df.shape)



# ==========================
# Check genes
# ==========================

print("\nChecking genes...")

missing = [
    g for g in GENES
    if g not in df.index
]


if missing:
    raise Exception(
        f"Missing genes: {missing}"
    )


print("All 8 genes found")



# ==========================
# Extract genes
# ==========================

print("\nExtracting 8 genes...")


gene_df = df.loc[GENES]


print("Before transpose:")
print(gene_df.shape)



# ==========================
# Samples x genes
# ==========================

combat_df = gene_df.T


print("\nAfter transpose:")
print(combat_df.shape)


print("\nColumns:")
print(combat_df.columns.tolist())



# ==========================
# Save
# ==========================

os.makedirs(
    "data",
    exist_ok=True
)


combat_df.to_csv(
    OUTPUT
)


print("""
==========================================
FINISHED

Saved:
data/GSE99039_8gene_combat.csv

==========================================
""")