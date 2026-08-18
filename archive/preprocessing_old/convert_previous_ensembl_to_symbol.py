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
# Load annotation file
# ==========================

annot = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)

print("\nAnnotation:")
print(annot.shape)

print(annot.columns.tolist())


# ==========================
# Merge Ensembl -> Symbol
# ==========================

merged = previous.merge(
    annot,
    left_on="Gene",
    right_on="_id",
    how="left"
)


print("\nAfter mapping:")
print(merged.shape)


# فقط ژن‌هایی که symbol دارند

mapped = merged[
    merged["symbol"].notna()
]


print("\nMapped genes:")
print(mapped.shape)

print(
    mapped[
        [
            "Gene_x",
            "symbol",
            "Coefficient_x"
        ]
    ].head(20)
)

# ==========================
# Save
# ==========================
mapped.to_csv(
    "data/previous_biomarkers_symbol.csv",
    index=False
)

print("\nSaved successfully")