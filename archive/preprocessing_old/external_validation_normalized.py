import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("="*60)
print("NORMALIZED EXTERNAL VALIDATION GSE165082")
print("="*60)


# ==========================
# Load model and scaler
# ==========================

model = joblib.load(
    "model/final_parkinson_logistic_model.pkl"
)

scaler = joblib.load(
    "model/final_scaler.pkl"
)


genes = open(
    "model/final_genes.txt"
).read().splitlines()



# ==========================
# Load external RNA-seq
# ==========================

ext = pd.read_csv(
    "data/GSE165082_gene_symbol.csv"
)


ext = ext.set_index(
    "GeneSymbol"
)



# ==========================
# CPM normalization
# ==========================

library_size = ext.sum(axis=0)


cpm = ext.div(
    library_size,
    axis=1
) * 1e6



log_cpm = np.log2(
    cpm + 1
)



# transpose samples x genes

X_ext = log_cpm.T



# ==========================
# Labels
# ==========================

labels=[]


for s in X_ext.index:

    if "PD" in s:
        labels.append(1)

    elif "CC" in s:
        labels.append(0)


y_ext=np.array(labels)



# ==========================
# Select genes
# ==========================

X_ext = X_ext[genes]



# ==========================
# Same scaler
# ==========================

X_scaled = scaler.transform(
    X_ext
)



# ==========================
# Prediction
# ==========================

prob = model.predict_proba(
    X_scaled
)[:,1]


pred = model.predict(
    X_scaled
)



# ==========================
# Results
# ==========================

print("\nProbability:")
print(prob)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_ext,
        pred
    )
)


print(
"\nAccuracy:",
accuracy_score(
    y_ext,
    pred
)
)


print(
"AUC:",
roc_auc_score(
    y_ext,
    prob
)
)


print(
classification_report(
    y_ext,
    pred
)
)


result=pd.DataFrame(
{
"Sample":X_ext.index,
"True_Label":y_ext,
"Prediction":pred,
"PD_probability":prob
}
)


result.to_csv(
"results/GSE165082_normalized_external_validation.csv",
index=False
)


print("\nSaved:")
print(
"results/GSE165082_normalized_external_validation.csv"
)