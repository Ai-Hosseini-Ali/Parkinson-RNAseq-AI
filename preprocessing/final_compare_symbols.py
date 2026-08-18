import pandas as pd


# ==========================
# Load new biomarkers
# ==========================

new = pd.read_csv(
    "data/GSE99039_candidate_biomarkers.csv"
)

print("New:")
print(new.shape)


# ==========================
# Load old annotated
# ==========================

old = pd.read_csv(
    "data/previous_with_annotation.csv"
)

print("Old:")
print(old.shape)


# ==========================
# Prepare symbols
# ==========================

new_symbols = set(
    new["Gene"]
    .astype(str)
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


# ==========================
# Intersection
# ==========================

shared = new_symbols.intersection(
    old_symbols
)


print("\nOverlap:")
print(len(shared))


# ==========================
# Extract shared genes
# ==========================

result = old[
    old["symbol"].isin(shared)
]


print("\nShared biomarkers:")
print(result)


result.to_csv(
    "data/final_shared_biomarkers.csv",
    index=False
)


print("\nSaved successfully")