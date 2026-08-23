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
# GSE136666 DEG overlap genes
# ==========================

genes = [
    "TH",
    "ADRA1D",
    "NTSR1",
    "DOK7",
    "OR7C1",
    "ABR",
    "CHST8",
    "TUBB3",
    "SLC6A3",
    "GABRD",
    "TRDN",
    "DRD2",
    "DDC",
    "CA7",
    "SLC38A2",
    "SOWAHA",
    "EN1",
    "ALDH1A1",
    "LINC01616",
    "SLC18A2",
    "P2RX6",
    "OR7A5",
    "AGTR1",
    "DNAH17-AS1",
    "SLC10A4",
    "ESM1"
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


# ==========================
# Select genes
# ==========================

missing = [
    g for g in genes
    if g not in expr.index
]


if missing:
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


label_column = "Label"


y = labels[label_column]


# هماهنگ کردن نمونه‌ها

common = (
    X.index
    .intersection(y.index)
)


X = X.loc[common]
y = y.loc[common]


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
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            penalty="l2",
            random_state=42
        )
    )
]),



"SVM":

Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        SVC(
            kernel="linear",
            probability=True,
            random_state=42
        )
    )
]),



"Random Forest":

Pipeline([
    (
        "model",
        RandomForestClassifier(
            n_estimators=500,
            random_state=42
        )
    )
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


results = []


print("\nResults:")


for name, model in models.items():

    acc = cross_val_score(
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


    results.append(
        {
            "Model": name,
            "Accuracy_mean": acc.mean(),
            "Accuracy_std": acc.std(),
            "AUC_mean": auc.mean(),
            "AUC_std": auc.std()
        }
    )


    print("\n", name)

    print(
        "Accuracy:",
        acc.mean(),
        "+/-",
        acc.std()
    )

    print(
        "AUC:",
        auc.mean(),
        "+/-",
        auc.std()
    )



# ==========================
# Save results
# ==========================

results_df = pd.DataFrame(results)


output = (
    "data/GEO/GSE136666/processed/"
    "GSE136666_GSE99039_26DEG_ML_results.csv"
)


results_df.to_csv(
    output,
    index=False
)


print("\nSaved:")
print(output)