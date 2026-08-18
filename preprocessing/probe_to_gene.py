import pandas as pd


# ==========================
# Load expression data
# ==========================

expr = pd.read_csv(
    "data/GSE99039_series_matrix.txt",
    sep="\t",
    comment="!"
)


print("Expression:")
print(expr.shape)


# حذف ستون‌های اضافی احتمالی
expr = expr.rename(columns={"ID_REF": "Probe"})


# ==========================
# Load GPL570 annotation
# ==========================

annot = pd.read_csv(
    "data/GPL570.annot",
    sep="\t",
    skiprows=27,
    low_memory=False
)


print("\nAnnotation:")
print(annot.shape)


print("\nColumns:")
print(annot.columns.tolist())


# ==========================
# Select required columns
# ==========================

annot = annot[
    [
        "ID",
        "Gene symbol"
    ]
]


annot.columns = [
    "Probe",
    "Gene"
]


print("\nAnnotation selected:")
print(annot.head())


# ==========================
# Merge Probe -> Gene
# ==========================

merged = expr.merge(
    annot,
    on="Probe",
    how="left"
)


print("\nMerged:")
print(merged.shape)


print(merged.head())


# ==========================
# Remove probes without gene
# ==========================

merged = merged.dropna(
    subset=["Gene"]
)


print("\nAfter removing unknown genes:")
print(merged.shape)


# ==========================
# Save
# ==========================

merged.to_csv(
    "data/GSE99039_probe_gene_expression.csv",
    index=False
)


print("\nSaved successfully")