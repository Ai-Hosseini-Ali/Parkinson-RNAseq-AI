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
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================
# PATHS
# ============================================================

dataset_file = "data/GSE99039_top100_dataset.csv"
signature_file = "data/robust_final_signature.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(dataset_file)
signature = pd.read_csv(signature_file)

print("\n==============================")
print("GSE99039 FINAL 15-GENE MODEL")
print("==============================")

print("\nDataset shape:")
print(df.shape)

print("\nClasses:")
print(df["Disease"].value_counts())


# ============================================================
# SELECT FINAL 15 GENES
# ============================================================

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
for i, gene in enumerate(genes, 1):
    print(f"{i:02d}. {gene}")


# ============================================================
# CHECK GENES
# ============================================================

available = [
    gene
    for gene in genes
    if gene in df.columns
]

missing = [
    gene
    for gene in genes
    if gene not in df.columns
]

print("\nAvailable genes:")
print(len(available))

print("\nMissing genes:")
print(missing)


if len(available) != 15:
    raise ValueError(
        f"Expected 15 genes, but only {len(available)} are available."
    )


# ============================================================
# FEATURES / LABEL
# ============================================================

X = df[available].copy()

y = (
    df["Disease"]
    .map({
        "CONTROL": 0,
        "IPD": 1
    })
)


print("\nX shape:")
print(X.shape)

print("\ny distribution:")
print(y.value_counts())


# ============================================================
# MODELS
# ============================================================

models = {

    "SVM": Pipeline([
        (
            "scale",
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

    "Logistic": Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

}


# ============================================================
# 10-FOLD STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

results = []

predictions = {}


for name, model in models.items():

    print("\n==============================")
    print("Running:", name)
    print("==============================")

    pred_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    pred = (
        pred_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y,
        pred_prob
    )

    accuracy = accuracy_score(
        y,
        pred
    )

    cm = confusion_matrix(
        y,
        pred
    )

    print("\nAUC:")
    print(auc)

    print("\nAccuracy:")
    print(accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    results.append([
        name,
        auc,
        accuracy
    ])

    predictions[name] = {
        "prob": pred_prob,
        "pred": pred
    }


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "AUC",
        "Accuracy"
    ]
)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(results_df)


results_df.to_csv(
    "data/final15_model_results.csv",
    index=False
)


# ============================================================
# TRAIN FINAL SVM ON ALL DATA
# ============================================================

final_model = models["SVM"]

final_model.fit(
    X,
    y
)

joblib.dump(
    final_model,
    "data/final_15gene_model.pkl"
)


# ============================================================
# SAVE FINAL GENES
# ============================================================

with open(
    "data/final_15_genes.txt",
    "w"
) as f:

    for gene in genes:
        f.write(gene + "\n")


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6, 5))

for name in predictions:

    prob = predictions[name]["prob"]

    fpr, tpr, _ = roc_curve(
        y,
        prob
    )

    auc = roc_auc_score(
        y,
        prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
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

plt.legend()

plt.tight_layout()

plt.savefig(
    "ROC_final15.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFUSION MATRIX - FINAL SVM
# ============================================================

svm_pred = predictions["SVM"]["pred"]

cm = confusion_matrix(
    y,
    svm_pred
)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["CONTROL", "IPD"],
    yticklabels=["CONTROL", "IPD"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Confusion Matrix - Final SVM"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_final15.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n==============================")
print("FINISHED SUCCESSFULLY")
print("==============================")

print("\nSaved files:")

print("data/final_15gene_model.pkl")
print("data/final15_model_results.csv")
print("data/final_15_genes.txt")
print("ROC_final15.png")
print("confusion_matrix_final15.png")