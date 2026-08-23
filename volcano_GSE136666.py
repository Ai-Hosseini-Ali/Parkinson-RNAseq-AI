import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Input
deg_file = "data/GEO/GSE136666/processed/GSE136666_DEG_results.csv"


# Load DEG results

deg = pd.read_csv(
    deg_file,
    index_col=0
)


print("DEG shape:")
print(deg.shape)


# Remove missing values

deg = deg.dropna(
    subset=["padj", "log2FoldChange"]
)


# DEG criteria

padj_threshold = 0.05
logfc_threshold = 1


deg["Significance"] = "Not significant"


deg.loc[
    (deg["padj"] < padj_threshold) &
    (deg["log2FoldChange"] > logfc_threshold),
    "Significance"
] = "Upregulated"


deg.loc[
    (deg["padj"] < padj_threshold) &
    (deg["log2FoldChange"] < -logfc_threshold),
    "Significance"
] = "Downregulated"


print("\nDEG counts:")
print(
    deg["Significance"].value_counts()
)


# Save significant genes

significant = deg[
    deg["Significance"] != "Not significant"
]


significant.to_csv(
    "data/GEO/GSE136666/processed/GSE136666_significant_DEGs.csv"
)


# Volcano plot

plt.figure(figsize=(9,7))


colors = {
    "Upregulated": "red",
    "Downregulated": "blue",
    "Not significant": "gray"
}


for group in colors:

    subset = deg[
        deg["Significance"] == group
    ]

    plt.scatter(
        subset["log2FoldChange"],
        -np.log10(subset["padj"]),
        s=15,
        label=group
    )


plt.axvline(
    logfc_threshold,
    linestyle="--"
)

plt.axvline(
    -logfc_threshold,
    linestyle="--"
)

plt.axhline(
    -np.log10(padj_threshold),
    linestyle="--"
)


plt.xlabel("log2 Fold Change")
plt.ylabel("-log10 adjusted p-value")

plt.title(
    "Volcano Plot - GSE136666 Parkinson vs Control"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "data/GEO/GSE136666/processed/GSE136666_Volcano.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nVolcano plot saved!")