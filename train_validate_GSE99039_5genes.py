import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Paths
# ==========================

expression_file = (
    "data/"
    "GSE99039_gene_expression.csv"
)

label_file = (
    "data/"
    "GSE99039_labels.csv"
)


# ==========================
# Genes from GSE136666
# ==========================

genes = [
    "DOK7",
    "CA7",
    "SOWAHA",
    "ADRA1D",
    "TRDN"
]


# ==========================
# Load expression
# ==========================

expr = pd.read_csv(
    expression_file,
    index_col=0
)


print("Expression:")
print(expr.shape)


# Select genes

missing = [
    g for g in genes
    if g not in expr.index
]

if len(missing) > 0:
    raise ValueError(
        f"Missing genes: {missing}"
    )


X = expr.loc[genes].T


print("\nFeature matrix:")
print(X.shape)


# ==========================
# Load labels
# ==========================

labels = pd.read_csv(
    label_file,
    index_col=0
)


print("\nLabels:")
print(labels.head())


# پیدا کردن ستون label
label_column = labels.columns[0]


y = labels[label_column]


# هماهنگ کردن نمونه‌ها

common_samples = (
    X.index
    .intersection(y.index)
)


X = X.loc[common_samples]
y = y.loc[common_samples]


print("\nFinal dataset:")
print(X.shape)

print("\nClass distribution:")
print(y.value_counts())


# ==========================
# Models
# ==========================

models = {

    "Logistic Regression":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model",
         LogisticRegression(
             max_iter=500,
             random_state=42
         ))
    ]),


    "SVM":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model",
         SVC(
             kernel="linear",
             probability=True,
             random_state=42
         ))
    ]),


    "Random Forest":
    Pipeline([
        ("model",
         RandomForestClassifier(
             n_estimators=500,
             random_state=42
         ))
    ])
}


# ==========================
# Cross Validation
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\nResults:")


for name, model in models.items():

    accuracy = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )


    auc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )


    print("\n", name)

    print(
        "Accuracy:",
        accuracy.mean(),
        "+/-",
        accuracy.std()
    )

    print(
        "AUC:",
        auc.mean(),
        "+/-",
        auc.std()
    )