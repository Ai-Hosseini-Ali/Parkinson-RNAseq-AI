import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import os


print("==============================")
print("GSE49036 EXTERNAL VALIDATION")
print("==============================")


BASE = "model"


model = joblib.load(
    os.path.join(BASE,"final_parkinson_logistic_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE,"final_scaler.pkl")
)


genes = open(
    os.path.join(BASE,"final_genes.txt")
).read().splitlines()


print("Signature genes:")
print(genes)



print("\nLoading GSE49036...")


data = pd.read_csv(
    "data/GSE49036_series_matrix.txt",
    sep="\t",
    comment="!",
    index_col=0
)


print("Raw:")
print(data.shape)

print(data.head())
