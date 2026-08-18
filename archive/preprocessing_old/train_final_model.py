import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)


print("==============================")
print("TRAIN FINAL PARKINSON MODEL")
print("==============================")


# =========================
# Final genes
# =========================

genes = [
    'PTGDS',
    'KIR2DL1',
    'KIR2DL3',
    'LILRB1',
    'KIAA0319L',
    'PPP4C',
    'TYROBP',
    'MBOAT7',
    'MMP9',
    'HLA-C',
    'LRRC25',
    'KIR3DL1',
    'BCL2',
    'RHOG'
]


# save genes

with open(
    "data/final_genes.txt",
    "w"
) as f:

    for g in genes:
        f.write(g+"\n")



# =========================
# Load training data
# =========================


expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)



if "Sample" not in expression.columns:

    expression.rename(
        columns={
            "Unnamed: 0":"Sample"
        },
        inplace=True
    )



data = expression.merge(
    labels,
    on="Sample"
)



print(
"Dataset:",
data.shape
)



X=data[genes]

y=data["Label"]



# =========================
# log transform
# =========================

X=np.log2(
    X+1
)



# =========================
# Scaling
# =========================

scaler=StandardScaler()


X_scaled=scaler.fit_transform(
    X
)



# =========================
# Train Logistic
# =========================


model=LogisticRegression(
    max_iter=5000,
    random_state=42
)


model.fit(
    X_scaled,
    y
)



# training evaluation

prob=model.predict_proba(
    X_scaled
)[:,1]


pred=model.predict(
    X_scaled
)



print(
"Training AUC:",
roc_auc_score(
    y,
    prob
)
)


print(
"Accuracy:",
accuracy_score(
    y,
    pred
)
)



# =========================
# Save model
# =========================


joblib.dump(
    model,
    "data/final_parkinson_logistic_model.pkl"
)


joblib.dump(
    scaler,
    "data/final_scaler.pkl"
)



print("\nSaved:")
print(
"final_parkinson_logistic_model.pkl"
)

print(
"final_scaler.pkl"
)

print(
"final_genes.txt"
)