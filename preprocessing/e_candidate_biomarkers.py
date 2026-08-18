import pandas as pd


# =========================
# Load Differential Expression
# =========================

de = pd.read_csv(
    "data/differential_expression.csv"
)

print("DE:")
print(de.shape)


# =========================
# Load FDR results
# =========================

de_fdr = pd.read_csv(
    "data/GSE99039_DE_FDR.csv"
)


# انتخاب ژن‌های معنی‌دار
sig = de_fdr[
    de_fdr["FDR"] < 0.05
]

print("\nSignificant genes:")
print(sig.shape)


# =========================
# Load ML importance
# =========================

ml = pd.read_csv(
    "data/GSE99039_feature_importance.csv"
)

print("\nML importance:")
print(ml.head())


# =========================
# Select top ML genes
# =========================

top_ml = ml.head(100)


# =========================
# Merge
# =========================

candidate = sig.merge(
    top_ml,
    on="Gene",
    how="inner"
)


print("\nCandidate biomarkers:")
print(candidate.shape)

print(candidate.head(20))


# =========================
# Save
# =========================

candidate.to_csv(
    "data/GSE99039_candidate_biomarkers.csv",
    index=False
)


print("\nSaved successfully")