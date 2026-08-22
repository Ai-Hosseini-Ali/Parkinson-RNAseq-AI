import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
TEST LOGISTIC
NORMALIZED GSE49036
8 GENE SIGNATURE
==========================================
""")


# ==========================
# Paths
# ==========================

DATA = "data/GSE49036_8gene_normalized.csv"

MODEL = "models/final_8gene_logistic.pkl"



# ==========================
# Load data
# ==========================

df = pd.read_csv(DATA)


print("Dataset:")
print(df.shape)



# ==========================
# Prepare X y
# ==========================

genes = [
    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"
]


X = df[genes]

y = df["Label"]



print("\nClasses:")
print(y.value_counts())



# ==========================
# Load model
# ==========================

model = joblib.load(
    MODEL
)



# ==========================
# Prediction
# ==========================

pred = model.predict(
    X
)


prob = model.predict_proba(
    X
)[:,1]



# ==========================
# Results
# ==========================


print("""
================================
LOGISTIC RESULTS
================================
""")


print(
    "Accuracy:",
    accuracy_score(
        y,
        pred
    )
)


print(
    "AUC:",
    roc_auc_score(
        y,
        prob
    )
)



print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y,
        pred
    )
)



print("\nClassification Report:")

print(
    classification_report(
        y,
        pred,
        zero_division=0
    )
)



print("""
==========================================
FINISHED
==========================================
""")