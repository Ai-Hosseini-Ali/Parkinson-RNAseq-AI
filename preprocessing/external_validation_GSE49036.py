import os
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
EXTERNAL VALIDATION
GSE49036
FINAL 8-GENE MODEL
==========================================
""")


DATA = "data/GSE49036_validation_8gene.csv"


MODEL_LOGISTIC = "models/final_8gene_logistic.pkl"
MODEL_SVM = "models/final_8gene_svm.pkl"
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



# Load data

df = pd.read_csv(DATA)


print("Dataset:")
print(df.shape)



X = df[GENES]

y = df["Label"]



print("\nClasses:")
print(y.value_counts())



# Load models

scaler = joblib.load(SCALER)

logistic = joblib.load(MODEL_LOGISTIC)

svm = joblib.load(MODEL_SVM)



X_scaled = scaler.transform(X)



def evaluate(model,name):

    pred = model.predict(X_scaled)


    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(X_scaled)[:,1]

    else:
        prob = model.decision_function(X_scaled)



    print("\n================================")
    print(name)
    print("================================")


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


    print("\nClassification Report:")
    print(
        classification_report(y,pred)
    )


    return {
        "Model":name,
        "Accuracy":accuracy_score(y,pred),
        "AUC":roc_auc_score(y,prob)
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



os.makedirs(
    "results",
    exist_ok=True
)


pd.DataFrame(results).to_csv(
    "results/GSE49036_external_validation.csv",
    index=False
)



print("""
==========================================
FINISHED

Saved:
results/GSE49036_external_validation.csv

==========================================
""")