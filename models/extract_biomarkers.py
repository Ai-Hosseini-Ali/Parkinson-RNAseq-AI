import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# Load data
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


# استخراج 500 ژن
mask = genes.isin(top_genes)

filtered = counts[mask]

selected_gene_names = genes[mask].values


# Samples x Genes
X = filtered.T

X.columns = selected_gene_names


# =========================
# Labels
# =========================

y = np.array([
    1 if "PD" in sample else 0
    for sample in X.index
])


# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# Train Logistic Regression
# =========================

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_scaled,
    y
)


# =========================
# Gene importance
# =========================

importance = pd.DataFrame({

    "Gene": X.columns,

    "Coefficient": model.coef_[0]

})


# قدرت اثر
importance["Abs_Coefficient"] = (
    importance["Coefficient"].abs()
)


importance = importance.sort_values(
    by="Abs_Coefficient",
    ascending=False
)


print("\nTop 20 Biomarker Genes:")
print(
    importance.head(20)
)


# ذخیره
importance.to_csv(
    "data/biomarker_genes.csv",
    index=False
)


print("\nSaved:")
print("data/biomarker_genes.csv")