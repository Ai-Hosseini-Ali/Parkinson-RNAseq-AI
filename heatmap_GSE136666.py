import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# Files

expression_file = "data/GEO/GSE136666/processed/GSE136666_log_expression.csv"
deg_file = "data/GEO/GSE136666/processed/GSE136666_DEG_final_annotation.csv"
metadata_file = "data/GEO/GSE136666/processed/GSE136666_metadata.csv"


output = "data/GEO/GSE136666/processed/GSE136666_DEG_heatmap.png"



# Load data

expr = pd.read_csv(
    expression_file,
    index_col=0
)

deg = pd.read_csv(
    deg_file
)

meta = pd.read_csv(
    metadata_file
)


print("Expression:")
print(expr.shape)

print("\nDEG:")
print(deg.shape)



# Select significant genes

genes = deg["Gene_Symbol"].dropna().tolist()


# Keep only genes موجود

genes = [
    g for g in genes
    if g in expr.index
]


print("\nGenes used:")
print(len(genes))


heatmap_data = expr.loc[genes]


# Z-score normalization

scaler = StandardScaler()

scaled = pd.DataFrame(
    scaler.fit_transform(heatmap_data.T).T,
    index=heatmap_data.index,
    columns=heatmap_data.columns
)



# Add sample annotation

colors = meta["Condition"].map(
    {
        "Control":0,
        "Parkinson":1
    }
)


# Plot

plt.figure(
    figsize=(10,14)
)


sns.clustermap(
    scaled,
    cmap="vlag",
    row_cluster=True,
    col_cluster=True,
    figsize=(10,14),
    yticklabels=True
)


plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)


print("\nSaved:")
print(output)