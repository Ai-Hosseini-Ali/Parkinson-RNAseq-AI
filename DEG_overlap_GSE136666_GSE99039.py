import pandas as pd
import os


# ==========================
# Paths
# ==========================

deg_file = (
    "data/GEO/GSE136666/processed/"
    "GSE136666_DEG_final_annotation.csv"
)

gse99039_file = (
    "data/"
    "GSE99039_gene_expression.csv"
)

output_file = (
    "data/GEO/GSE136666/processed/"
    "GSE136666_DEG_GSE99039_overlap.csv"
)


# ==========================
# Load GSE136666 DEG
# ==========================

if not os.path.exists(deg_file):
    raise FileNotFoundError(
        "GSE136666 DEG file not found:\n"
        + deg_file
    )


deg = pd.read_csv(deg_file)


print("GSE136666 DEG:")
print(deg.shape)

print("\nColumns:")
print(deg.columns.tolist())


# ==========================
# Select significant genes
# ==========================

# فقط ژن‌های معنی‌دار
deg_sig = deg[
    deg["Significance"].isin(
        [
            "Upregulated",
            "Downregulated"
        ]
    )
]


print("\nSignificant DEG:")
print(deg_sig.shape)


# گرفتن Gene Symbol

deg_genes = (
    deg_sig["Gene_Symbol"]
    .dropna()
    .unique()
    .tolist()
)


print("\nDEG genes:")
print(len(deg_genes))
print(deg_genes)



# ==========================
# Load GSE99039
# ==========================

if not os.path.exists(gse99039_file):
    raise FileNotFoundError(
        "GSE99039 expression file not found:\n"
        + gse99039_file
    )


expr99039 = pd.read_csv(
    gse99039_file,
    index_col=0
)


print("\nGSE99039 expression:")
print(expr99039.shape)



# ==========================
# Find overlap
# ==========================

gse99039_genes = set(
    expr99039.index
)


found = [
    gene for gene in deg_genes
    if gene in gse99039_genes
]


missing = [
    gene for gene in deg_genes
    if gene not in gse99039_genes
]


print("\nGenes available in GSE99039:")
print(len(found))
print(found)


print("\nMissing genes:")
print(len(missing))
print(missing)



# ==========================
# Save result
# ==========================

overlap = deg_sig[
    deg_sig["Gene_Symbol"].isin(found)
]


overlap.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)