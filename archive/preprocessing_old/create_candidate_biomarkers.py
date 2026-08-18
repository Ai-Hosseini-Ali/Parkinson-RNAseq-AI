import pandas as pd

# ==========================
# Load DE
# ==========================

de = pd.read_csv(
    "data/GSE99039_DE_FDR.csv"
)

print("DE:")
print(de.shape)

print(de.columns)


# ==========================
# Significant genes
# ==========================

sig = de[
    de["FDR"] < 0.05
]

print("\nSignificant genes:")
print(sig.shape)


# ==========================
# Feature importance
# ==========================

imp = pd.read_csv(
    "data/GSE99039_feature_importance.csv"
)

print("\nFeature Importance:")
print(imp.shape)


# ==========================
# Merge
# ==========================

candidate = pd.merge(
    sig,
    imp,
    on="Gene",
    how="inner"
)


candidate = candidate.sort_values(
    "Importance",
    ascending=False
)


print("\nCandidate Biomarkers:")
print(candidate.shape)

print(candidate.head(20))


# Save

candidate.to_csv(
    "data/GSE99039_candidate_biomarkers.csv",
    index=False
)

print("\nSaved successfully")