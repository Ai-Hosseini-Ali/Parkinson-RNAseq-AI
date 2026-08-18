import pandas as pd

# load files

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


# normalize scores

ml["ML_score"] = (
    ml["Importance"] /
    ml["Importance"].max()
)
import numpy as np

de["DE_score"] = (
    -np.log10(de["FDR"])
)

de["DE_score"] = (
    de["DE_score"] /
    de["DE_score"].max()
)

ppi["PPI_score"] = (
    ppi["Degree"] /
    ppi["Degree"].max()
)


stab["Stability_score"] = (
    stab["Stability_%"] /
    100
)


# merge

df = (
    ml[["Gene","ML_score"]]
    .merge(
        de[["Gene","DE_score"]],
        on="Gene",
        how="inner"
    )
    .merge(
        ppi[["Gene","PPI_score"]],
        on="Gene",
        how="left"
    )
    .merge(
        stab[["Gene","Stability_score"]],
        on="Gene",
        how="left"
    )
)


df = df.fillna(0)


# final score

df["Robust_score"] = (
    0.25*df["ML_score"]
    +
    0.25*df["DE_score"]
    +
    0.25*df["PPI_score"]
    +
    0.25*df["Stability_score"]
)


df = df.sort_values(
    "Robust_score",
    ascending=False
)


print("\nROBUST FINAL SIGNATURE\n")

print(
    df.head(20)
)


df.head(50).to_csv(
    "data/robust_final_signature.csv",
    index=False
)


print("\nSaved:")
print(
    "data/robust_final_signature.csv"
)