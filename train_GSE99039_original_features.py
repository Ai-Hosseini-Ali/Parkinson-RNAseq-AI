import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    make_scorer
)


# ==========================
# Paths
# ==========================

expression_file = (
    "data/GSE99039_gene_expression.csv"
)

label_file = (
    "data/GSE99039_labels.csv"
)


# ==========================
# Load expression
# ==========================

expr = pd.read_csv(
    expression_file,
    index_col=0
)

print("Expression:")
print(expr.shape)


# ==========================
# Load labels
# ==========================

labels = pd.read_csv(
    label_file,
    index_col=0
)


print("\nLabels:")
print(labels.head())


# ==========================
# Align samples
# ==========================

common_samples = (
    expr.columns.intersection(labels.index)
)


expr = expr[
    common_samples
]


labels = labels.loc[
    common_samples
]


y = labels["Label"]


print("\nFinal dataset:")
print(expr.shape)


print("\nClass distribution:")
print(y.value_counts())


# ==========================
# Feature matrix
# ==========================

X = expr.T


print("\nFeature matrix:")
print(X.shape)



# ==========================
# Models
# ==========================


models = {


"Logistic Regression":

Pipeline([
    ("scaler", StandardScaler()),
    ("model",
     LogisticRegression(
         max_iter=5000,
         penalty="l1",
         solver="liblinear"
     ))
]),



"SVM":

Pipeline([
    ("scaler", StandardScaler()),
    ("model",
     SVC(
         kernel="linear",
         probability=True
     ))
]),



"Random Forest":

RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

}



# ==========================
# Cross validation
# ==========================


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {

"accuracy":
"accuracy",

"auc":
"roc_auc",

"f1":
"f1"

}



print("\nResults:\n")


for name, model in models.items():

    result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )


    print("="*40)
    print(name)

    print(
        "Accuracy:",
        np.mean(result["test_accuracy"]),
        "+/-",
        np.std(result["test_accuracy"])
    )


    print(
        "AUC:",
        np.mean(result["test_auc"]),
        "+/-",
        np.std(result["test_auc"])
    )


    print(
        "F1:",
        np.mean(result["test_f1"]),
        "+/-",
        np.std(result["test_f1"])
    )
