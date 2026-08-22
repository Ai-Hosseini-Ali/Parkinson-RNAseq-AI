import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
EXTERNAL VALIDATION
GSE165082
FINAL 8-GENE SIGNATURE
==========================================
""")


# ==========================
# Paths
# ==========================

DATA_PATH = "data/GSE165082_gene_symbol.csv"
LABEL_PATH = "data/GSE165082_labels.csv"

MODEL_LOGISTIC = "models/final_8gene_logistic.pkl"
MODEL_SVM = "models/final_8gene_svm.pkl"
SCALER_PATH = "models/final_8gene_scaler.pkl"


# ==========================
# Final genes
# ==========================

GENES = [
    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"
]


# ==========================
# Load expression
# ==========================

print("Loading expression data...")


expr = pd.read_csv(DATA_PATH)


print("\nRaw:")
print(expr.shape)


# ==========================
# Gene orientation
# ==========================

expr = expr.set_index("GeneSymbol")


# Remove duplicate genes

expr = expr[~expr.index.duplicated(keep="first")]


# transpose
# samples become rows

X = expr.T


print("\nAfter transpose:")
print(X.shape)



# ==========================
# Load labels
# ==========================

labels = pd.read_csv(
    LABEL_PATH
)


print("\nLabels:")
print(labels.head())


# sample names alignment

labels = labels.set_index("Sample")


common = X.index.intersection(
    labels.index
)


X = X.loc[common]

labels = labels.loc[common]


y = labels["Label"]


print("\nClasses:")
print(y.value_counts())



# ==========================
# Select genes
# ==========================

missing = [
    g for g in GENES
    if g not in X.columns
]


if missing:

    print("\nMissing genes:")
    print(missing)

    raise Exception(
        "Some genes missing"
    )



X = X[GENES]


print("\nSelected genes:")

for g in GENES:
    print(g)



print("\nFinal matrix:")
print(X.shape)



# ==========================
# Load models
# ==========================


print("\nLoading models...")


scaler = joblib.load(
    SCALER_PATH
)


logistic = joblib.load(
    MODEL_LOGISTIC
)


svm = joblib.load(
    MODEL_SVM
)



# ==========================
# Scale
# ==========================


X_scaled = scaler.transform(
    X
)



# ==========================
# Evaluation
# ==========================


def evaluate(model,name):


    pred = model.predict(
        X_scaled
    )


    if hasattr(model,"predict_proba"):

        prob = model.predict_proba(
            X_scaled
        )[:,1]

    else:

        prob = model.decision_function(
            X_scaled
        )


    print("""
================================
{}
================================
""".format(name))


    print(
        "Accuracy:",
        accuracy_score(y,pred)
    )


    print(
        "AUC:",
        roc_auc_score(y,prob)
    )


    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y,pred
        )
    )


    print("\nReport:")
    print(
        classification_report(
            y,pred
        )
    )


    return {

        "Model":name,

        "Accuracy":
        accuracy_score(y,pred),

        "AUC":
        roc_auc_score(y,prob)

    }




results=[]


results.append(
    evaluate(
        logistic,
        "Logistic Regression"
    )
)


results.append(
    evaluate(
        svm,
        "SVM"
    )
)



# ==========================
# Save
# ==========================


os.makedirs(
    "results",
    exist_ok=True
)


pd.DataFrame(results).to_csv(
    "results/external_validation_8gene.csv",
    index=False
)



print("""
==========================================
FINISHED SUCCESSFULLY

Saved:

results/external_validation_8gene.csv

==========================================
""")