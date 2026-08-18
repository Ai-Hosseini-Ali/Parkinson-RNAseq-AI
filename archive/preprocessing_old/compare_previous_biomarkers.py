import pandas as pd


# ==========================
# Load files
# ==========================

current = pd.read_csv(
    "data/GSE99039_candidate_biomarkers.csv"
)

previous = pd.read_csv(
    "data/biomarker_genes.csv"
)


# ==========================
# Normalize gene names
# ==========================

def clean_gene(x):
    return (
        str(x)
        .upper()
        .strip()
        .split("///")[0]
        .split(";")[0]
    )


current["Gene_clean"] = current["Gene"].apply(clean_gene)

previous["Gene_clean"] = previous["Gene"].apply(clean_gene)


# ==========================
# Intersection
# ==========================

overlap = set(current["Gene_clean"]) & set(previous["Gene_clean"])


print("Current genes:")
print(len(current))


print("Previous genes:")
print(len(previous))


print("Overlap:")
print(len(overlap))


print("\nShared genes:")

print(
    current[
        current["Gene_clean"].isin(overlap)
    ][
        [
            "Gene",
            "LogFC",
            "FDR",
            "Importance"
        ]
    ]
)


# ==========================
# Save
# ==========================

result = current.copy()

result["Validation"] = result["Gene_clean"].apply(
    lambda x:
    "Previous_project"
    if x in overlap
    else
    "New_candidate"
)


result.to_csv(
    "data/GSE99039_validated_biomarkers_v2.csv",
    index=False
)

print("\nSaved")