import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


# paths
expression_file = "data/GEO/GSE136666/processed/GSE136666_log_expression.csv"
deg_file = "data/GEO/GSE136666/processed/GSE136666_feature_ranking.csv"
metadata_file = "data/GEO/GSE136666/processed/GSE136666_metadata.csv"


# load data
expr = pd.read_csv(expression_file, index_col=0)

ranking = pd.read_csv(deg_file)

metadata = pd.read_csv(metadata_file)


print("Expression:")
print(expr.shape)


# take top DEG genes
top_genes = ranking.head(43)["Gene"].tolist()


# keep existing genes
genes = [g for g in top_genes if g in expr.index]


print("\nGenes used:")
print(len(genes))


X = expr.loc[genes].T


# labels
y = metadata["Condition"].map({
    "Control":0,
    "Parkinson":1
})


print("\nFeature matrix:")
print(X.shape)


# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



selection_count = pd.Series(
    0,
    index=genes
)


# repeated CV

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for repeat in range(100):

    for train_idx, test_idx in cv.split(X_scaled,y):

        X_train = X_scaled[train_idx]
        y_train = y.iloc[train_idx]


        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=500
        )


        model.fit(
            X_train,
            y_train
        )


        coef = pd.Series(
            abs(model.coef_[0]),
            index=genes
        )


        selected = coef[coef>0].index


        selection_count[selected]+=1



result = pd.DataFrame({
    "Gene":selection_count.index,
    "Selections":selection_count.values
})


result["Frequency_%"] = (
    result["Selections"] /
    result["Selections"].max()
    *100
)


result = result.sort_values(
    "Frequency_%",
    ascending=False
)


print("\nTop stable genes:")
print(result.head(20))


output = (
"data/GEO/GSE136666/processed/"
"GSE136666_stability_selection.csv"
)


result.to_csv(
    output,
    index=False
)


print("\nSaved:")
print(output)