import pandas as pd
import numpy as np
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


print("=" * 60)
print("TRAIN COMBAT HARMONIZED PARKINSON MODEL")
print("=" * 60)


# ==========================
# Load locked genes
# ==========================

with open(
    "pipeline/gene_signature_locked.json",
    encoding="utf-8"
) as f:
    genes = json.load(f)["genes"]


# ==========================
# Load harmonized data
# ==========================

data = pd.read_csv(
    "data/combat_harmonized_14genes.csv",
    index_col=0
)


print("\nDataset:")
print(data.shape)


# ==========================
# Separate datasets
# ==========================

train = data.iloc[:438]
external = data.iloc[438:]


print("\nTrain:")
print(train.shape)

print("External:")
print(external.shape)



# ==========================
# Labels
# ==========================

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


labels = labels[
    labels["Disease"].isin(
        ["CONTROL", "IPD"]
    )
]


y_train = (
    labels["Disease"]
    .map(
        {
            "CONTROL":0,
            "IPD":1
        }
    )
    .values
)


# GSE165082 labels

external_labels = pd.read_csv(
    "data/GSE165082_labels.csv"
)


y_external = external_labels["Label"].values



# ==========================
# Train model
# ==========================

X_train = train[genes]

X_external = external[genes]


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_external_scaled = scaler.transform(
    X_external
)



model = LogisticRegression(
    max_iter=5000,
    random_state=42
)


model.fit(
    X_train_scaled,
    y_train
)



# ==========================
# Internal performance
# ==========================

train_prob = model.predict_proba(
    X_train_scaled
)[:,1]


print("\nInternal training:")
print(
    "AUC:",
    roc_auc_score(
        y_train,
        train_prob
    )
)



# ==========================
# External validation
# ==========================

ext_prob = model.predict_proba(
    X_external_scaled
)[:,1]


ext_pred = model.predict(
    X_external_scaled
)


print("\nExternal validation:")
print(
    "AUC:",
    roc_auc_score(
        y_external,
        ext_prob
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_external,
        ext_pred
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_external,
        ext_pred
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_external,
        ext_pred
    )
)



# ==========================
# Save model
# ==========================

joblib.dump(
    model,
    "model/combat_logistic_model.pkl"
)

joblib.dump(
    scaler,
    "model/combat_scaler.pkl"
)


print("\nSaved:")
print("model/combat_logistic_model.pkl")
print("model/combat_scaler.pkl")