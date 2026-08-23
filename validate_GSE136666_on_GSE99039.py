import pandas as pd
import os


# ==========================
# Paths
# ==========================

stability_file = (
    "data/GEO/GSE136666/processed/"
    "GSE136666_stability_selection.csv"
)

# Correct GSE99039 path
gse99039_file = (
    "data/"
    "GSE99039_gene_expression.csv"
)

output_file = (
    "data/GEO/GSE136666/processed/"
    "GSE136666_GSE99039_overlap.csv"
)


# ==========================
# Load stable genes
# ==========================

stability = pd.read_csv(stability_file)

print("Stability genes:")
print(stability.head())


# Select stable genes
stable_genes = stability[
    stability["Frequency_%"] >= 30
]["Gene"].tolist()


print("\nStable genes:")
print(len(stable_genes))
print(stable_genes)



# ==========================
# Load GSE99039 expression
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
# Find overlapping genes
# ==========================

available_genes = set(expr99039.index)


found_genes = [
    gene for gene in stable_genes
    if gene in available_genes
]


missing_genes = [
    gene for gene in stable_genes
    if gene not in available_genes
]


print("\nFound in GSE99039:")
print(len(found_genes))
print(found_genes)


print("\nMissing from GSE99039:")
print(len(missing_genes))
print(missing_genes)



# ==========================
# Save overlap result
# ==========================

result = pd.DataFrame({
    "Gene": stable_genes,
    "Available_in_GSE99039": [
        gene in available_genes
        for gene in stable_genes
    ]
})


result.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)