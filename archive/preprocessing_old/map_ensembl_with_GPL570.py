import pandas as pd


# ==========================
# Load previous biomarkers
# ==========================

previous = pd.read_csv(
    "data/biomarker_genes.csv"
)

print("Previous:")
print(previous.shape)


# ==========================
# Load GPL570 annotation
# ==========================

annot = pd.read_csv(
    "data/GPL570.annot",
    sep="\t",
    skiprows=27,
    dtype=str
)

print("\nAnnotation:")
print(annot.shape)

print(annot.columns[:5])


# ==========================
# Select needed columns
# ==========================

annot = annot[
    [
        "Gene ID",
        "Gene symbol"
    ]
]


# ==========================
# Clean
# ==========================

annot = annot.dropna()

annot["Gene ID"] = annot["Gene ID"].str.strip()

annot["Gene symbol"] = annot["Gene symbol"].str.strip()


# ==========================
# Merge
# ==========================

merged = previous.merge(
    annot,
    left_on="Gene",
    right_on="Gene ID",
    how="left"
)


print("\nAfter mapping:")
print(merged.shape)


mapped = merged.dropna(
    subset=["Gene symbol"]
)


print("\nMapped:")
print(mapped.shape)


print(
    mapped.head(20)
)


# ==========================
# Save
# ==========================

mapped.to_csv(
    "data/previous_biomarkers_symbol.csv",
    index=False
)


print("\nSaved successfully")