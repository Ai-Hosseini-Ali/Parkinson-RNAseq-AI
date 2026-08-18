import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================
# Load expression
# ==========================

data = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


# Fix sample column

data = data.rename(
    columns={
        "Unnamed: 0": "Sample"
    }
)


# ==========================
# Load labels
# ==========================

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


data = data.merge(
    labels[["Sample","Label"]],
    on="Sample"
)


print("Dataset:")
print(data.shape)



# ==========================
# Load biomarkers
# ==========================

biomarkers = pd.read_csv(
    "data/final_ranked_biomarkers.csv"
)


genes = biomarkers.head(20)["Gene"].tolist()



genes = [
    g for g in genes
    if g in data.columns
]


print("\nGenes used:")
print(genes)



# ==========================
# X y
# ==========================

X = data[genes]

y = data["Label"]



# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# ==========================
# Model
# ==========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)



# ==========================
# 5 Fold CV
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



prob = cross_val_predict(
    model,
    X_scaled,
    y,
    cv=cv,
    method="predict_proba"
)[:,1]



pred = (
    prob >= 0.5
).astype(int)



# ==========================
# Results
# ==========================

auc = roc_auc_score(
    y,
    prob
)


accuracy = accuracy_score(
    y,
    pred
)



print("\n===================")
print("Cross Validation Results")
print("===================")

print(
    "AUC:",
    round(auc,4)
)

print(
    "Accuracy:",
    round(accuracy,4)
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
        pred
    )
)



# Save predictions

out = pd.DataFrame({

    "Sample": data["Sample"],

    "True_Label": y,

    "Prediction": pred,

    "Probability": prob

})


out.to_csv(
    "data/cv_predictions.csv",
    index=False
)


print("\nSaved:")
print("data/cv_predictions.csv")