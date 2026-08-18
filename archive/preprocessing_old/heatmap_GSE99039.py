import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load dataset
# ===========================

data = pd.read_csv("data/GSE99039_top100_dataset.csv")

print("Dataset:")
print(data.shape)

# ===========================
# Separate labels
# ===========================

labels = data["Disease"]

X = data.drop(columns=["Disease"])

# ===========================
# Select Top20 genes
# ===========================

top20 = X.iloc[:, :20]

print("\nTop20 genes:")
print(top20.columns.tolist())

# ===========================
# Create heatmap dataframe
# ===========================

heatmap_data = top20.T

# ===========================
# Color labels
# ===========================

colors = labels.map({
    "CONTROL": "blue",
    "IPD": "red"
})

# ===========================
# Plot
# ===========================

plt.figure(figsize=(16,8))

sns.clustermap(
    heatmap_data,
    cmap="RdBu_r",
    z_score=0,
    col_colors=colors,
    figsize=(16,10)
)

plt.savefig(
    "data/GSE99039_heatmap_top20.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nHeatmap saved successfully.")