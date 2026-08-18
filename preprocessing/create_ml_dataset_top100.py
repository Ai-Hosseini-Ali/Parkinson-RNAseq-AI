import pandas as pd

# ==========================
# Load full dataset
# ==========================

data = pd.read_csv("data/GSE99039_final_dataset.csv")

print("Dataset:")
print(data.shape)

# ==========================
# Load Top100 genes
# ==========================

top = pd.read_csv("top100_genes_GSE99039.csv")

genes = top["Gene"].tolist()

print("Top genes:", len(genes))

# ==========================
# Keep existing genes only
# ==========================

existing = [g for g in genes if g in data.columns]

print("Genes found:", len(existing))

# ==========================
# Create ML dataset
# ==========================

cols = existing + ["Disease"]

ml = data[cols]

print("\nML dataset:")
print(ml.shape)

print(ml["Disease"].value_counts())

# ==========================
# Save
# ==========================

ml.to_csv(
    "data/GSE99039_top100_dataset.csv",
    index=False
)

print("\nSaved successfully.")