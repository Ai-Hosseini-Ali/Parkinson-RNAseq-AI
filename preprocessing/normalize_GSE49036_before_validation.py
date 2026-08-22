import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
NORMALIZE GSE49036
8 GENE SIGNATURE
==========================================
""")


# Paths

DATA = "data/GSE49036_validation_8gene.csv"
MODEL = "models/final_8gene_svm.pkl"
SCALER = "models/final_8gene_scaler.pkl"


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
# Load validation
# ==========================

df = pd.read_csv(DATA)

print("Dataset:")
print(df.shape)


print("\nColumns:")
print(df.columns)



# ==========================
# Separate X y
# ==========================

y = df["Label"]

X = df[GENES]


print("\nOriginal classes:")
print(y.value_counts())


# ==========================
# Normalization
# ==========================

print("""
Applying validation normalization...
""")


scaler_validation = StandardScaler()


X_norm = scaler_validation.fit_transform(X)


X_norm = pd.DataFrame(
    X_norm,
    columns=GENES
)



# ==========================
# Load trained model
# ==========================

model = joblib.load(MODEL)


# ==========================
# Prediction
# ==========================

pred = model.predict(
    X_norm
)


prob = model.decision_function(
    X_norm
)


print("""
================================
RESULT
================================
""")


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
    confusion_matrix(y,pred)
)


print("\nReport:")
print(
    classification_report(
        y,
        pred,
        zero_division=0
    )
)


# Save normalized data

os.makedirs(
    "data",
    exist_ok=True
)


X_norm["Label"] = y.values


X_norm.to_csv(
    "data/GSE49036_8gene_normalized.csv",
    index=False
)


print("""
==========================================
FINISHED

Saved:
data/GSE49036_8gene_normalized.csv

==========================================
""")