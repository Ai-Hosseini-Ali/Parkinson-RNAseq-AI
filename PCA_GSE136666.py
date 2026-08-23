import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Paths
expression_file = "data/GEO/GSE136666/processed/GSE136666_log_expression.csv"
metadata_file = "data/GEO/GSE136666/processed/GSE136666_metadata.csv"


# Load expression
expr = pd.read_csv(
    expression_file,
    index_col=0
)

# Load metadata
meta = pd.read_csv(metadata_file)


print("Expression shape:")
print(expr.shape)

print("\nMetadata:")
print(meta)


# Transpose
# samples become rows
X = expr.T


# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame(
    data=principal_components,
    columns=["PC1","PC2"]
)


pca_df["Sample"] = meta["Sample"]
pca_df["Condition"] = meta["Condition"]


print("\nExplained variance:")
print(pca.explained_variance_ratio_)


# Save PCA coordinates

pca_df.to_csv(
    "data/GEO/GSE136666/processed/GSE136666_PCA_coordinates.csv",
    index=False
)


# Plot

plt.figure(figsize=(8,6))


for condition in pca_df["Condition"].unique():

    subset = pca_df[
        pca_df["Condition"] == condition
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=100,
        label=condition
    )


    for _, row in subset.iterrows():
        plt.text(
            row["PC1"],
            row["PC2"],
            row["Sample"].split("_")[-1],
            fontsize=8
        )


plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)"
)

plt.title("PCA - GSE136666 Substantia Nigra")

plt.legend()

plt.grid(True)


plt.savefig(
    "data/GEO/GSE136666/processed/GSE136666_PCA.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nPCA completed!")