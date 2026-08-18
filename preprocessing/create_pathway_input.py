import pandas as pd

# =========================
# Load new biomarkers
# =========================

new = pd.read_csv(
    "data/GSE99039_candidate_biomarkers.csv"
)

new = new[["Gene", "LogFC", "FDR", "Importance"]]

new["Source"] = "GSE99039"


# =========================
# Load previous biomarkers
# =========================

old = pd.read_csv(
    "data/previous_biomarkers_merged.csv"
)

old = old[
    old["symbol"].notna()
]

old = old[
    ["symbol", "Coefficient"]
]


old.columns = [
    "Gene",
    "Importance"
]

old["LogFC"] = None
old["FDR"] = None
old["Source"] = "Previous"


# =========================
# Merge
# =========================

combined = pd.concat(
    [
        new,
        old
    ],
    ignore_index=True
)


print("Total genes:")
print(combined.shape)

print("\nSources:")
print(combined["Source"].value_counts())


# =========================
# Save
# =========================

combined.to_csv(
    "data/pathway_input_biomarkers.csv",
    index=False
)

print("\nSaved successfully")