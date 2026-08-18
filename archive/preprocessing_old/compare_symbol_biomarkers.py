import pandas as pd


# ======================
# Load new biomarkers
# ======================

new = pd.read_csv(
    "data/GSE99039_candidate_biomarkers.csv"
)

print("New:")
print(new.shape)


# ======================
# Load previous annotated
# ======================

old = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)

print("Previous:")
print(old.shape)


print(old.columns)


# ======================
# Extract symbols
# ======================

new_symbols = set(
    new["Gene"].astype(str)
)


old_symbols = set(
    old["symbol"]
    .dropna()
    .astype(str)
)


print("\nNew symbols:")
print(len(new_symbols))


print("Old symbols:")
print(len(old_symbols))


# ======================
# Intersection
# ======================

shared = new_symbols.intersection(old_symbols)


print("\nOverlap:")
print(len(shared))


result = new[
    new["Gene"].isin(shared)
]


print("\nShared biomarkers:")
print(result)


# ======================
# Save
# ======================

result.to_csv(
    "data/shared_previous_current_biomarkers.csv",
    index=False
)


print("\nSaved successfully")