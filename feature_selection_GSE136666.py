import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Files

expression_file = "data/GEO/GSE136666/processed/GSE136666_log_expression.csv"
deg_file = "data/GEO/GSE136666/processed/GSE136666_DEG_final_annotation.csv"
metadata_file = "data/GEO/GSE136666/processed/GSE136666_metadata.csv"

output_file = "data/GEO/GSE136666/processed/GSE136666_feature_ranking.csv"



# Load

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


# DEG genes

genes = (
    deg["Gene_Symbol"]
    .dropna()
    .unique()
    .tolist()
)


genes = [
    g for g in genes
    if g in expr.index
]


print("\nGenes used:")
print(len(genes))


# Matrix

X = expr.loc[genes].T


# Labels

y = meta["Condition"].map(
    {
        "Control":0,
        "Parkinson":1
    }
)


print("\nFeature matrix:")
print(X.shape)

print("\nLabels:")
print(y.value_counts())



# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



results = pd.DataFrame(
    {
        "Gene": genes
    }
)



# 1 - ANOVA

anova_score, anova_p = f_classif(
    X_scaled,
    y
)


results["ANOVA_score"] = anova_score
results["ANOVA_pvalue"] = anova_p



# 2 - Mutual Information

mi = mutual_info_classif(
    X_scaled,
    y,
    random_state=42
)


results["Mutual_information"] = mi



# 3 - LASSO Logistic Regression

lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    random_state=42,
    max_iter=5000
)


lasso.fit(
    X_scaled,
    y
)


results["LASSO_coefficient"] = (
    lasso.coef_[0]
)



# Ranking

results["LASSO_abs"] = (
    results["LASSO_coefficient"]
    .abs()
)


results = results.sort_values(
    by="LASSO_abs",
    ascending=False
)



print("\nTop genes:")

print(
    results.head(15)
)



# Save

results.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)