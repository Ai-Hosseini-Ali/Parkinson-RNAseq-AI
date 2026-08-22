import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
OPTIMIZE THRESHOLD
GSE49036
SVM 8-GENE MODEL
==========================================
""")


# ==========================
# Paths
# ==========================

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
# Load data
# ==========================

df = pd.read_csv(DATA)


X = df[GENES]

y = df["Label"]


print("Dataset:")
print(df.shape)


print("\nClasses:")
print(y.value_counts())



# ==========================
# Load model
# ==========================

scaler = joblib.load(SCALER)

model = joblib.load(MODEL)



X_scaled = scaler.transform(X)



# ==========================
# Get probability score
# ==========================

if hasattr(model, "predict_proba"):

    score = model.predict_proba(
        X_scaled
    )[:,1]

else:

    score = model.decision_function(
        X_scaled
    )

    # normalize scores
    score = (
        score-score.min()
    ) / (
        score.max()-score.min()
    )



print("\nAUC:")
print(
    roc_auc_score(y,score)
)



# ==========================
# Threshold search
# ==========================

results=[]


for t in np.arange(
    0.1,
    0.91,
    0.05
):

    pred = (
        score >= t
    ).astype(int)


    acc = accuracy_score(
        y,
        pred
    )


    tn,fp,fn,tp = confusion_matrix(
        y,
        pred
    ).ravel()


    sensitivity = tp/(tp+fn)

    specificity = tn/(tn+fp)



    results.append({

        "Threshold":round(t,2),

        "Accuracy":acc,

        "Sensitivity":sensitivity,

        "Specificity":specificity

    })



result = pd.DataFrame(results)



print("""
================================
Threshold Results
================================
""")


print(result)



best = result.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]



print("""
================================
Best Threshold
================================
""")


print(best)



# ==========================
# Evaluate best
# ==========================

threshold = best["Threshold"]


pred = (
    score >= threshold
).astype(int)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y,
        pred
    )
)


print("\nReport:")
print(
    classification_report(
        y,
        pred,
        zero_division=0
    )
)



result.to_csv(
    "results/GSE49036_threshold_analysis.csv",
    index=False
)


print("""
==========================================
FINISHED

Saved:
results/GSE49036_threshold_analysis.csv

==========================================
""")