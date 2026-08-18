import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# =========================
# Load expression data
# =========================

df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)

genes = df["Geneid"]
counts = df.drop("Geneid", axis=1)


# =========================
# Load selected genes
# =========================

selected = pd.read_csv(
    "data/top500_genes.csv"
)

top_genes = selected["Gene"].tolist()

filtered = counts[genes.isin(top_genes)]

print(filtered.shape)

# =========================
# Samples x Genes
# =========================

X = filtered.T

# =========================
# Labels
# =========================

y = []

for sample in X.index:

    if "PD" in sample:
        y.append(1)

    elif "CC" in sample:
        y.append(0)

y = np.array(y)

print(X.shape)
print(y.shape)

# =========================
# MLP Model
# =========================

model = Pipeline([
    ("scaler", StandardScaler()),

    ("mlp",
        MLPClassifier(
            hidden_layer_sizes=(32,16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=5000,
            random_state=42
        )
    )
])

# =========================
# Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nAccuracy scores:")
print(scores)

print("\nMean Accuracy:")
print(scores.mean())

print("\nStd:")
print(scores.std())