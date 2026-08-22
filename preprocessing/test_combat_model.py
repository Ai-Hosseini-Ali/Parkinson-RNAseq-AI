import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
TEST COMBAT CORRECTED DATA
GSE99039 + GSE49036
8 GENE SIGNATURE
==========================================
""")


DATA = "data/combat_corrected_8gene.csv"


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
# Load
# ==========================

print("Loading ComBat data...")

df = pd.read_csv(
    DATA,
    index_col=0
)


print("Shape:")
print(df.shape)



# ==========================
# Split
# ==========================

# first 558 = GSE99039
# last 23 = GSE49036

train = df.iloc[:, :558]
valid = df.iloc[:, 558:]


print("\nTraining:")
print(train.shape)


print("\nValidation:")
print(valid.shape)



# ==========================
# Labels
# ==========================

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


y_train = labels["Label"].values


valid_labels = pd.read_csv(
    "data/GSE49036_labels.csv"
)


y_valid = valid_labels["Label"].values



print("\nClasses training:")
print(pd.Series(y_train).value_counts())


print("\nClasses validation:")
print(pd.Series(y_valid).value_counts())



# ==========================
# Prepare
# ==========================

X_train = train.T
X_valid = valid.T


print("\nX train:")
print(X_train.shape)

print("X valid:")
print(X_valid.shape)



# ==========================
# Scaling
# ==========================

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_valid = scaler.transform(
    X_valid
)



# ==========================
# Models
# ==========================

models = {

"SVM":
SVC(
    kernel="linear",
    probability=True,
    random_state=42
),


"Logistic":
LogisticRegression(
    max_iter=1000,
    random_state=42
)

}



# ==========================
# Test
# ==========================


for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")


    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_valid
    )


    prob = model.predict_proba(
        X_valid
    )[:,1]


    print(
        "Accuracy:",
        accuracy_score(
            y_valid,
            pred
        )
    )


    print(
        "AUC:",
        roc_auc_score(
            y_valid,
            prob
        )
    )


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


print("""
==========================================
FINISHED
==========================================
""")