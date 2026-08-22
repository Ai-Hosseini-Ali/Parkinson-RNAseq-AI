import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


print("="*50)
print("TEST COMBAT MODELS")
print("GSE99039 + GSE49036")
print("8 GENE SIGNATURE")
print("="*50)


DATA = "data/combat_corrected_8gene.csv"

TRAIN_LABELS = "data/GSE99039_labels.csv"
VALID_LABELS = "data/GSE49036_labels.csv"



print("\nLoading ComBat data...")

combat = pd.read_csv(
    DATA,
    index_col=0
)


print("Original:")
print(combat.shape)


# genes x samples -> samples x genes

combat = combat.T


print("After transpose:")
print(combat.shape)



print("\nLoading labels...")

train_labels = pd.read_csv(
    TRAIN_LABELS
)


valid_labels = pd.read_csv(
    VALID_LABELS
)



# ==========================
# Match samples
# ==========================

train_ids = train_labels["Sample"]

valid_ids = valid_labels["Sample"]


X_train = combat.loc[
    combat.index.intersection(train_ids)
]


X_valid = combat.loc[
    combat.index.intersection(valid_ids)
]


label_train = train_labels.set_index(
    "Sample"
)

label_valid = valid_labels.set_index(
    "Sample"
)


y_train = label_train.loc[
    X_train.index,
    "Label"
]


y_valid = label_valid.loc[
    X_valid.index,
    "Label"
]


print("\nTraining:")
print(X_train.shape)


print("Validation:")
print(X_valid.shape)


print("\nTraining classes:")
print(y_train.value_counts())


print("\nValidation classes:")
print(y_valid.value_counts())



# ==========================
# Models
# ==========================


models = {

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


    "Logistic":
    Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),


    "RandomForest":
    RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )
}



results=[]


plt.figure(
    figsize=(7,6)
)



for name, model in models.items():

    print("\n")
    print("="*40)
    print(name)
    print("="*40)


    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_valid
    )


    if hasattr(
        model,
        "predict_proba"
    ):
        prob = model.predict_proba(
            X_valid
        )[:,1]

    else:
        prob = model.decision_function(
            X_valid
        )


    acc = accuracy_score(
        y_valid,
        pred
    )


    auc = roc_auc_score(
        y_valid,
        prob
    )


    print("Accuracy:", acc)
    print("AUC:", auc)


    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_valid,
            pred
        )
    )


    print("\nReport:")
    print(
        classification_report(
            y_valid,
            pred
        )
    )


    results.append(
        [
            name,
            acc,
            auc
        ]
    )


    RocCurveDisplay.from_predictions(
        y_valid,
        prob,
        name=name
    )



os.makedirs(
    "results",
    exist_ok=True
)


pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC"
    ]
).to_csv(
    "results/combat_model_results.csv",
    index=False
)



plt.title(
    "ROC - ComBat 8 Gene"
)


plt.savefig(
    "results/combat_ROC.png",
    dpi=300
)



print("\n")
print("="*50)
print("FINISHED")
print("Saved:")
print("results/combat_model_results.csv")
print("results/combat_ROC.png")
print("="*50)