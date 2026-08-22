import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("""
==========================================
FINAL 8-GENE MODEL TRAINING
GSE99039
==========================================
""")


# ==========================
# Paths
# ==========================

DATA_PATH = "data/GSE99039_top100_dataset.csv"

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# ==========================
# Final 8 genes
# ==========================

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
# Load Dataset
# ==========================

print("Loading dataset...")


df = pd.read_csv(
    DATA_PATH
)


print("\nDataset:")
print(df.shape)


print("\nFirst columns:")
print(df.columns[:10])



# ==========================
# Split X and y
# ==========================

if "Disease" not in df.columns:

    raise Exception(
        "Disease column not found!"
    )


y = df["Disease"].copy()


X = df.drop(
    columns=["Disease"]
)



print("\nOriginal classes:")
print(y.value_counts())



# ==========================
# Encode labels
# ==========================

label_map = {

    "CONTROL":0,
    "Control":0,
    "control":0,

    "IPD":1,
    "PD":1,
    "Parkinson":1

}


y = y.map(label_map)



# Remove unknown labels

if y.isna().sum() > 0:

    print(
        "Unknown labels found:"
    )

    print(
        df.loc[y.isna(),"Disease"]
    )

    raise Exception(
        "Label conversion failed"
    )


# Force integer

y = y.astype(int)



print("\nEncoded classes:")
print(y.value_counts())


print("\nLabel dtype:")
print(y.dtype)



# ==========================
# Select genes
# ==========================

missing = [
    g for g in GENES
    if g not in X.columns
]


if missing:

    raise Exception(
        f"Missing genes: {missing}"
    )


X = X[GENES]



print("""
Selected 8 genes:
""")


for g in GENES:
    print(g)



# ==========================
# Clean data
# ==========================

X = X.astype(float)


X = X.replace(
    [np.inf,-np.inf],
    np.nan
)


X = X.dropna()


y = y.loc[X.index]



print("\nFinal matrix:")
print(X.shape)



# ==========================
# Scaling
# ==========================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)



# ==========================
# Models
# ==========================


logistic = LogisticRegression(
    max_iter=5000,
    random_state=42
)



svm = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)



# ==========================
# Train Logistic
# ==========================


print("""
Training Logistic Regression...
""")


logistic.fit(
    X_scaled,
    y
)



# ==========================
# Train SVM
# ==========================


print("""
Training SVM...
""")


svm.fit(
    X_scaled,
    y
)



# ==========================
# Evaluation
# ==========================


for name,model in [

    ("Logistic Regression",logistic),

    ("SVM",svm)

]:


    pred = model.predict(
        X_scaled
    )


    prob = model.predict_proba(
        X_scaled
    )[:,1]



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
        confusion_matrix(
            y,
            pred
        )
    )


    print("\nReport:")
    print(
        classification_report(
            y,
            pred
        )
    )



# ==========================
# Save
# ==========================


joblib.dump(
    logistic,
    f"{MODEL_DIR}/final_8gene_logistic.pkl"
)


joblib.dump(
    svm,
    f"{MODEL_DIR}/final_8gene_svm.pkl"
)


joblib.dump(
    scaler,
    f"{MODEL_DIR}/final_8gene_scaler.pkl"
)



with open(
    f"{MODEL_DIR}/final_8gene_list.txt",
    "w"
) as f:

    for g in GENES:
        f.write(g+"\n")



print("""
==========================================
FINISHED SUCCESSFULLY

Saved:

models/
 ├── final_8gene_logistic.pkl
 ├── final_8gene_svm.pkl
 ├── final_8gene_scaler.pkl
 └── final_8gene_list.txt

==========================================
""")