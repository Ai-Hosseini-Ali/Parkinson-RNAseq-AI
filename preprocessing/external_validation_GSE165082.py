import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib


# ==================================
# Genes from robust final signature
# ==================================

genes = [
    'PTGDS',
    'KIR2DL1',
    'DGKK',
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


# ==================================
# Load GSE165082 external dataset
# ==================================

print("Loading GSE165082")

ext = pd.read_csv(
    "data/GSE165082_gene_symbol.csv"
)


ext = ext.set_index(
    "GeneSymbol"
)


# transpose
ext = ext.T


print("External:")
print(ext.shape)


# ==================================
# Create labels from sample names
# ==================================

labels=[]

for sample in ext.index:

    if "PD" in sample:
        labels.append(1)

    elif "CC" in sample:
        labels.append(0)

    else:
        labels.append(np.nan)


y_ext = pd.Series(
    labels,
    index=ext.index
)


mask = ~y_ext.isna()


ext = ext.loc[mask]
y_ext = y_ext.loc[mask]


print("\nExternal classes:")
print(y_ext.value_counts())


# ==================================
# Load training dataset
# ==================================

print("\nLoading GSE99039")
print("\nLoading GSE99039")


train = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


# پیدا کردن ستون نمونه

if "Sample" in train.columns:

    sample_col = "Sample"

elif "Unnamed: 0" in train.columns:

    train = train.rename(
        columns={"Unnamed: 0":"Sample"}
    )

    sample_col="Sample"

else:

    raise Exception(
        "No Sample column found"
    )


print(train.head())


train = train.merge(
    labels,
    on="Sample"
)


print(
"Training merged:",
train.shape
)

X_train = train[genes]

y_train = train["Label"]


# ==================================
# Common genes
# ==================================

available = [
    g for g in genes
    if g in ext.columns
]


print("\nGenes used:")
print(available)


# train and external only common genes

X_train = X_train[available]

X_ext = ext[available]


print(
"\nFeature number:",
len(available)
)


# ==================================
# Normalize
# ==================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_ext_scaled = scaler.transform(
    X_ext
)



# ==================================
# Train final model
# ==================================

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)


model.fit(
    X_train_scaled,
    y_train
)


# ==================================
# External prediction
# ==================================

prob = model.predict_proba(
    X_ext_scaled
)[:,1]


pred = model.predict(
    X_ext_scaled
)


# ==================================
# Results
# ==================================

auc = roc_auc_score(
    y_ext,
    prob
)


acc = accuracy_score(
    y_ext,
    pred
)


print("\n======================")
print("External Validation")
print("======================")


print(
"AUC:",
auc
)


print(
"Accuracy:",
acc
)


print("\nConfusion Matrix")

print(
confusion_matrix(
    y_ext,
    pred
)
)


print(
classification_report(
    y_ext,
    pred
)
)



# save

result = pd.DataFrame(
{
"Sample":ext.index,
"True_Label":y_ext,
"Prediction":pred,
"PD_probability":prob
}
)


result.to_csv(
"data/GSE165082_external_validation_results.csv",
index=False
)


print("\nSaved:")
print(
"data/GSE165082_external_validation_results.csv"
)