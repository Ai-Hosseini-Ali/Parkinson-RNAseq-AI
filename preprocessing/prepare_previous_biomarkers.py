import pandas as pd

# ==========================
# Load previous annotated genes
# ==========================

old = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)

print("Loaded:")
print(old.shape)

print(old.columns)


# ==========================
# Keep useful columns
# ==========================

previous = old[
    [
        "Gene",
        "Coefficient",
        "Abs_Coefficient",
        "log2FC",
        "p_value",
        "FDR",
        "symbol"
    ]
].copy()


# remove empty symbols

previous = previous[
    previous["symbol"].notna()
]


# uppercase symbols

previous["symbol"] = (
    previous["symbol"]
    .astype(str)
    .str.upper()
)


print("\nAfter cleaning:")
print(previous.shape)


# ==========================
# Save
# ==========================

previous.to_csv(
    "data/previous_biomarkers_merged.csv",
    index=False
)


print("\nSaved successfully")