import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


print("\n===================================")
print(" FINAL EXTERNAL VALIDATION ")
print(" GSE165082")
print("===================================\n")


# =====================================
# Locked 14 Gene Signature
# =====================================

GENES = [
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


# =====================================
# Paths
# =====================================

BASE = r"C:\Users\alihoseini\Desktop\Parkinson_AI\data"


TRAIN_X = os.path.join(
    BASE,
    "GSE99039_final_dataset.csv"
)

TRAIN_Y = os.path.join(
    BASE,
    "GSE99039_labels.csv"
)


EXT_X = os.path.join(
    BASE,
    "GSE165082_normalized_log_cpm.csv"
)

EXT_Y = os.path.join(
    BASE,
    "GSE165082_labels.csv"
)



print("Training file:")
print(TRAIN_X)

print("\nExternal file:")
print(EXT_X)



# =====================================
# Load data
# =====================================


print("\nLoading training data...")


train = pd.read_csv(
    TRAIN_X,
    index_col=0
)


train_labels = pd.read_csv(
    TRAIN_Y
)



print("Training shape:")
print(train.shape)



print("\nLoading external data...")


external = pd.read_csv(
    EXT_X,
    index_col=0
)


external_labels = pd.read_csv(
    EXT_Y
)



print("External shape:")
print(external.shape)



# =====================================
# Fix gene orientation
# =====================================


def prepare_gene_matrix(df):

    print("\nChecking gene format...")

    # genes as columns
    if all(g in df.columns for g in GENES):
        print("Genes detected in columns")
        return df

    # genes as index
    elif all(g in df.index for g in GENES):
        print("Genes detected in index")
        return df.T

    else:
        print("\nAvailable examples:")
        print(df.columns[:10])
        print(df.index[:10])

        missing = set(GENES) - set(df.columns)

        if len(missing) > 0:
            missing = set(GENES) - set(df.index)

        raise Exception(
            f"Missing genes: {missing}"
        )



train = prepare_gene_matrix(train)

external = prepare_gene_matrix(external)



# =====================================
# Check genes
# =====================================


print("\nChecking locked genes...")


missing_train = set(GENES) - set(train.columns)

missing_external = set(GENES) - set(external.columns)


print("Missing train:")
print(missing_train)


print("Missing external:")
print(missing_external)



if missing_train:
    raise Exception(
        "Training genes missing"
    )


if missing_external:
    raise Exception(
        "External genes missing"
    )



# =====================================
# Prepare X Y
# =====================================


X_train = train[GENES]

X_external = external[GENES]


y_train = train_labels["Label"]

y_external = external_labels["Label"]



print("\nFinal matrices:")

print(
    "Train:",
    X_train.shape
)

print(
    "External:",
    X_external.shape
)



# =====================================
# Scaling
# =====================================


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_external_scaled = scaler.transform(
    X_external
)



# =====================================
# Train Model
# =====================================


print("\nTraining Logistic Regression...")


model = LogisticRegression(
    max_iter=5000,
    random_state=42
)


model.fit(
    X_train_scaled,
    y_train
)



# =====================================
# Prediction
# =====================================


prediction = model.predict(
    X_external_scaled
)


probability = model.predict_proba(
    X_external_scaled
)[:,1]



# =====================================
# Evaluation
# =====================================


accuracy = accuracy_score(
    y_external,
    prediction
)


auc = roc_auc_score(
    y_external,
    probability
)


cm = confusion_matrix(
    y_external,
    prediction
)



print("\n===================================")
print(" EXTERNAL VALIDATION RESULT")
print("===================================")


print(
    "Accuracy:",
    accuracy
)


print(
    "AUC:",
    auc
)


print(
    "\nConfusion Matrix:"
)

print(cm)



# =====================================
# Save
# =====================================


os.makedirs(
    "results",
    exist_ok=True
)


result = pd.DataFrame({

    "Dataset":[
        "GSE165082"
    ],

    "Genes":[
        14
    ],

    "Accuracy":[
        accuracy
    ],

    "AUC":[
        auc
    ]

})


result.to_csv(
    "results/external_validation_final.csv",
    index=False
)



print("\nSaved:")
print(
    "results/external_validation_final.csv"
)