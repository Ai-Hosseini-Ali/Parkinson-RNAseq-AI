import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_score


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


# نگه داشتن فقط 500 ژن
filtered = counts[genes.isin(top_genes)]


print("Selected genes matrix:")
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


print("\nSamples:")
print(X.shape)

print("\nLabels:")
print(y)


# =========================
# Model
# =========================

model = Pipeline([
    ("scaler", StandardScaler()),

    ("logistic",
     LogisticRegression(
         max_iter=5000
     ))
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