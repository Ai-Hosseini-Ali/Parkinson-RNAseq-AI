import pandas as pd
from statsmodels.stats.multitest import multipletests


# ==========================
# Load DE
# ==========================

df = pd.read_csv(
    "data/GSE99039_differential_expression.csv"
)

print("Original:")
print(df.shape)


# ==========================
# FDR correction
# ==========================

_, fdr, _, _ = multipletests(
    df["PValue"],
    alpha=0.05,
    method="fdr_bh"
)

df["FDR"] = fdr


# ==========================
# Sort
# ==========================

df = df.sort_values(
    "FDR"
)


print("\nTop genes:")
print(df.head(10))


print("\nSignificant genes:")
print(
    (df["FDR"] < 0.05).sum()
)


# ==========================
# Save
# ==========================

df.to_csv(
    "data/GSE99039_DE_FDR.csv",
    index=False
)

print("\nSaved:")
print("data/GSE99039_DE_FDR.csv")