import pandas as pd
import gzip
import os


print("""
================================
ANNOTATE GSE99039
GPL570
================================
""")


DATA = "data/GSE99039_series_matrix.txt"
ANNOT = "data/GPL570.annot.gz"

OUTPUT = "data/GSE99039_gene_expression.csv"


# ==========================
# Load expression
# ==========================

print("\nLoading expression...")

expr = pd.read_csv(
    DATA,
    sep="\t",
    comment="!"
)

expr = expr.rename(
    columns={expr.columns[0]: "ID"}
)


print("Expression:")
print(expr.shape)

print("\nColumns:")
print(expr.columns[:5])



# ==========================
# Find annotation header
# ==========================

print("\nFinding annotation header...")

header_line = None

with gzip.open(
    ANNOT,
    "rt",
    encoding="utf-8",
    errors="ignore"
) as f:

    for i, line in enumerate(f):

        if line.startswith("ID"):
            header_line = i
            break


print("Header:")
print(header_line)



# ==========================
# Load annotation
# ==========================

print("\nLoading annotation...")


annot = pd.read_csv(
    ANNOT,
    sep="\t",
    skiprows=header_line,
    compression="gzip",
    low_memory=False
)


print("Annotation:")
print(annot.shape)



# ==========================
# Merge
# ==========================

print("\nMerging annotation...")


merged = expr.merge(
    annot[["ID","Gene symbol"]],
    on="ID",
    how="inner"
)


print("Merged:")
print(merged.shape)



# ==========================
# Remove empty genes
# ==========================

merged = merged[
    merged["Gene symbol"].notna()
]


merged = merged[
    merged["Gene symbol"] != ""
]



# ==========================
# Probe -> Gene
# ==========================

print("\nConverting probes to genes...")


gene_matrix = (
    merged
    .groupby("Gene symbol")
    .mean(numeric_only=True)
)



print("\nGene matrix:")
print(gene_matrix.shape)


# ==========================
# Check signature genes
# ==========================

genes = [
    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"
]


print("\nChecking genes:")
print("--------------------------------")


found = 0

for g in genes:

    if g in gene_matrix.index:
        print(g, "FOUND")
        found += 1

    else:
        print(g, "MISSING")


print("\nFound:")
print(found, "/", len(genes))



# ==========================
# Save
# ==========================

os.makedirs(
    "data",
    exist_ok=True
)


gene_matrix.to_csv(
    OUTPUT
)


print("""
================================
FINISHED

Saved:
data/GSE99039_gene_expression.csv

================================
""")