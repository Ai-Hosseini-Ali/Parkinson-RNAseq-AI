import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================
# Paths
# ============================

expression_file = "data/GSE99039_final_dataset.csv"
labels_file = "data/GSE99039_labels.csv"
signature_file = "data/robust_final_signature.csv"


# ============================
# Load data
# ============================

expression = pd.read_csv(expression_file)
labels = pd.read_csv(labels_file)
signature = pd.read_csv(signature_file)


print("Expression:")
print(expression.shape)

print("\nLabels:")
print(labels.head())


# ============================
# Fix sample column
# ============================

if "Sample" not in expression.columns:
    expression = expression.rename(
        columns={expression.columns[0]: "Sample"}
    )


# Merge labels

data = expression.merge(
    labels[["Sample", "Label"]],
    on="Sample"
)


print("\nMerged:")
print(data.shape)


# ============================
# Select 15 genes
# ============================

genes = (
    signature
    .sort_values(
        "Robust_score",
        ascending=False
    )
    .head(15)["Gene"]
    .tolist()
)


print("\nFinal 15 genes:")
print(genes)


available = [
    g for g in genes
    if g in data.columns
]


print("\nAvailable genes:")
print(available)



X = data[available]

y = data["Label"]


print("\nClasses:")
print(y.value_counts())


# ============================
# Models
# ============================

models = {

"SVM":
Pipeline([
    ("scale", StandardScaler()),
    ("model",
     SVC(
        kernel="linear",
        probability=True,
        random_state=42
     ))
]),


"Logistic":
Pipeline([
    ("scale", StandardScaler()),
    ("model",
     LogisticRegression(
        max_iter=2000,
        random_state=42
     ))
])

}


# ============================
# Cross validation
# ============================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


results=[]


for name, model in models.items():

    print("\nRunning:", name)

    pred_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:,1]


    pred = (
        pred_prob >= 0.5
    ).astype(int)


    auc = roc_auc_score(
        y,
        pred_prob
    )


    acc = accuracy_score(
        y,
        pred
    )


    print("AUC:", auc)
    print("Accuracy:", acc)


    print(
        confusion_matrix(
            y,
            pred
        )
    )


    results.append(
        [
            name,
            auc,
            acc
        ]
    )


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "AUC",
        "Accuracy"
    ]
)


print("\nResults:")
print(results_df)


results_df.to_csv(
    "data/final15_model_results.csv",
    index=False
)


# ============================
# Train final SVM
# ============================

final_model = models["SVM"]


final_model.fit(
    X,
    y
)


joblib.dump(
    final_model,
    "data/final_15gene_model.pkl"
)


# ============================
# ROC curve
# ============================

prob = cross_val_predict(
    final_model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:,1]


fpr,tpr,_ = roc_curve(
    y,
    prob
)


plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC - Final 15 Gene Signature"
)

plt.savefig(
    "ROC_final15.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ============================
# Confusion Matrix
# ============================

pred = (
    prob>=0.5
).astype(int)


cm = confusion_matrix(
    y,
    pred
)


plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "confusion_matrix_final15.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("\n===================")
print("FINISHED")
print("===================")

print("Saved:")
print("data/final_15gene_model.pkl")
print("data/final15_model_results.csv")
print("ROC_final15.png")
print("confusion_matrix_final15.png")