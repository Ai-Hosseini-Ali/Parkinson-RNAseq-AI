import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load data

df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)


genes = df["Geneid"]

X = df.drop(
    "Geneid",
    axis=1
)


# Filtering

X = X[X.sum(axis=1) >= 10]


# log normalization

X = np.log2(X + 1)


# samples as rows

X = X.T


# labels

labels = [
    1 if "PD" in x else 0
    for x in X.index
]


# scaling

X_scaled = StandardScaler().fit_transform(X)


# PCA

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)


# plot

plt.figure(figsize=(7,6))


for i, label in enumerate(labels):

    if label == 1:
        plt.scatter(
            components[i,0],
            components[i,1],
            label="PD"
        )
    else:
        plt.scatter(
            components[i,0],
            components[i,1],
            label="Control"
        )


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Parkinson RNA-seq Samples")

plt.legend()

plt.savefig(
    "data/figure1_PCA.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()