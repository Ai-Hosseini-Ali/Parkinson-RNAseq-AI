import pandas as pd


# ==========================
# Load previous genes
# ==========================

old = pd.read_csv(
    "data/candidate_genes.csv"
)

print("Previous:")
print(old.shape)


# ==========================
# Load annotation
# ==========================

annot = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)

print("Annotation:")
print(annot.shape)

print(annot.columns)


# ==========================
# Merge using Ensembl
# ==========================

merged = old.merge(
    annot,
    left_on="Gene",
    right_on="Gene",
    how="left"
)


print("\nMerged:")
print(merged.shape)


print(
    merged.head()
)


# ==========================
# Save
# ==========================

merged.to_csv(
    "data/previous_with_annotation.csv",
    index=False
)


print("Saved")