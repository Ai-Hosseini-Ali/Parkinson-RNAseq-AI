import pandas as pd


# ==============================
# Load new biomarkers
# ==============================

new = pd.read_csv(
    "data/GSE99039_candidate_biomarkers.csv"
)

print("New:")
print(new.shape)


# ==============================
# Load previous annotated genes
# ==============================

old = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)

print("Old:")
print(old.shape)

print(old.columns)


# ==============================
# Prepare columns
# ==============================

# New biomarkers
new_genes = set(
    new["Gene"]
    .astype(str)
)


# Previous genes
# Gene ID column from annotation

old_entrez = old[
    old["Gene ID"].notna()
].copy()


old_entrez["Gene ID"] = (
    old_entrez["Gene ID"]
    .astype(str)
)


old_genes = set(
    old_entrez["Gene ID"]
)


print("\nNew genes:")
print(len(new_genes))

print("Old Entrez:")
print(len(old_genes))


# ==============================
# Compare
# ==============================

shared = new_genes.intersection(old_genes)


print("\nOverlap:")
print(len(shared))


# ==============================
# Save
# ==============================

result = old_entrez[
    old_entrez["Gene ID"].isin(shared)
]


result.to_csv(
    "data/shared_entrez_biomarkers.csv",
    index=False
)


print("\nShared:")
print(result.head())


print("\nSaved successfully")