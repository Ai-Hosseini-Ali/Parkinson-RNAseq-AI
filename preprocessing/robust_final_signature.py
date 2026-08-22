import pandas as pd
import numpy as np


# ==========================
# Load files
# ==========================

ml = pd.read_csv(
    "data/GSE99039_feature_importance.csv"
)

de = pd.read_csv(
    "data/GSE99039_DE_FDR.csv"
)

ppi = pd.read_csv(
    "data/string_hub_genes.csv"
)

stab = pd.read_csv(
    "data/gene_stability_results.csv"
)


# ==========================
# Normalize ML score
# ==========================

ml["ML_score"] = (
    ml["Importance"] /
    ml["Importance"].max()
)


# ==========================
# Normalize DE score
# ==========================

de["DE_score"] = (
    -np.log10(
        de["FDR"].clip(lower=1e-300)
    )
)

de["DE_score"] = (
    de["DE_score"] /
    de["DE_score"].max()
)


# ==========================
# Normalize PPI score
# ==========================

ppi["PPI_score"] = (
    ppi["Degree"] /
    ppi["Degree"].max()
)


# ==========================
# Normalize Stability score
# ==========================

stab["Stability_score"] = (
    stab["Stability_%"] /
    100
)


# ==========================
# Merge all evidence
# ==========================

df = (
    ml[["Gene", "ML_score"]]
    .merge(
        de[["Gene", "DE_score"]],
        on="Gene",
        how="inner"
    )
    .merge(
        ppi[["Gene", "PPI_score"]],
        on="Gene",
        how="left"
    )
    .merge(
        stab[["Gene", "Stability_score"]],
        on="Gene",
        how="left"
    )
)


# Missing evidence = 0

df = df.fillna(0)


# ==========================
# Robust score
# ==========================

df["Robust_score"] = (
    0.25 * df["ML_score"] +
    0.25 * df["DE_score"] +
    0.25 * df["PPI_score"] +
    0.25 * df["Stability_score"]
)


# ==========================
# Rank genes
# ==========================

df = df.sort_values(
    "Robust_score",
    ascending=False
).reset_index(drop=True)


# ==========================
# Print results
# ==========================

print("\n==============================")
print("ROBUST FINAL SIGNATURE")
print("==============================")

print(
    df.head(20).to_string(index=False)
)


# ==========================
# Save top 50
# ==========================

df.head(50).to_csv(
    "data/robust_final_signature.csv",
    index=False
)


print("\nSaved:")
print("data/robust_final_signature.csv")