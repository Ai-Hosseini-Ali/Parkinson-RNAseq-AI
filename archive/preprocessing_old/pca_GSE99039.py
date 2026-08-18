import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================
# Load dataset
# ==========================

df = pd.read_csv(
    "data/GSE99039_final_dataset.csv",
    index_col=0
)

print("Dataset:")
print(df.shape)

# ==========================
# Split X and y
# ==========================

X = df.drop(columns=["Disease"])

y = df["Disease"]

print("\nFeatures:", X.shape)
print("Labels:", y.shape)

# ==========================
# Standardization
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardization done.")

# ==========================
# PCA
# ==========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance:")

print("PC1 =", round(pca.explained_variance_ratio_[0]*100,2), "%")

print("PC2 =", round(pca.explained_variance_ratio_[1]*100,2), "%")

# ==========================
# Plot
# ==========================

plt.figure(figsize=(8,6))

colors = {
    "CONTROL":"blue",
    "IPD":"red"
}

for disease in colors:

    idx = y==disease

    plt.scatter(
        X_pca[idx,0],
        X_pca[idx,1],
        s=40,
        alpha=0.7,
        label=disease,
        c=colors[disease]
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)"
)

plt.title("PCA - GSE99039")

plt.legend()

plt.tight_layout()

plt.savefig(
    "figure_GSE99039_PCA.png",
    dpi=300
)

plt.show()

print("\nFigure saved.")