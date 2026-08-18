import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


print("=" * 60)
print(" CROSS PLATFORM NORMALIZATION COMPARISON ")
print(" GSE99039 -> GSE165082 ")
print("=" * 60)


# =========================
# Paths
# =========================

TRAIN_FILE = Path(
    "data/GSE99039_final_dataset.csv"
)

EXTERNAL_FILE = Path(
    "data/GSE165082_normalized_log_cpm.csv"
)

TRAIN_LABEL_FILE = Path(
    "data/GSE99039_labels.csv"
)

EXTERNAL_LABEL_FILE = Path(
    "data/GSE165082_labels.csv"
)

GENE_FILE = Path(
    "model/final_genes.txt"
)

OUTPUT = Path(
    "results/cross_platform_normalization_comparison.csv"
)


# =========================
# Load locked genes
# =========================

genes = [
    g.strip()
    for g in open(GENE_FILE)
    if g.strip()
]


print("\nLocked genes:")
print(genes)


# =========================
# Load datasets
# =========================

train = pd.read_csv(TRAIN_FILE)

external = pd.read_csv(EXTERNAL_FILE)


train_labels = pd.read_csv(
    TRAIN_LABEL_FILE
)["Label"]


external_labels = pd.read_csv(
    EXTERNAL_LABEL_FILE
)["Label"]


print("\nTraining:")
print(train.shape)

print("External:")
print(external.shape)


print("\nExternal labels:")
print(external_labels.value_counts())


# =========================
# Select genes
# =========================

X_train = train[genes].astype(float)

X_external = external[genes].astype(float)


print("\nFinal matrices:")
print(
    "Train:",
    X_train.shape
)

print(
    "External:",
    X_external.shape
)



# =========================
# Normalization methods
# =========================

def train_based_zscore():

    scaler = StandardScaler()

    Xt = scaler.fit_transform(
        X_train
    )

    Xe = scaler.transform(
        X_external
    )

    return Xt, Xe



def self_scaled():

    scaler_train = StandardScaler()
    scaler_external = StandardScaler()


    Xt = scaler_train.fit_transform(
        X_train
    )

    Xe = scaler_external.fit_transform(
        X_external
    )

    return Xt, Xe



def rank_normalization():

    Xt = X_train.rank(
        axis=0,
        pct=True
    )

    Xe = X_external.rank(
        axis=0,
        pct=True
    )

    return Xt.values, Xe.values



methods = {

    "train_based_zscore":
        train_based_zscore(),

    "self_scaled":
        self_scaled(),

    "rank_normalization":
        rank_normalization()

}



# =========================
# Train and evaluate
# =========================

results = []


for name, data in methods.items():

    print("\n" + "="*40)
    print("Testing:", name)
    print("="*40)


    Xt, Xe = data


    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


    model.fit(
        Xt,
        train_labels
    )


    pred = model.predict(
        Xe
    )


    prob = model.predict_proba(
        Xe
    )[:,1]


    acc = accuracy_score(
        external_labels,
        pred
    )


    auc = roc_auc_score(
        external_labels,
        prob
    )


    cm = confusion_matrix(
        external_labels,
        pred
    )


    print(
        "Accuracy:",
        acc
    )


    print(
        "AUC:",
        auc
    )


    print(
        "Confusion Matrix:"
    )

    print(cm)



    results.append({

        "Method": name,

        "Accuracy": acc,

        "AUC": auc,

        "TN": cm[0,0],

        "FP": cm[0,1],

        "FN": cm[1,0],

        "TP": cm[1,1]

    })



# =========================
# Save results
# =========================

df = pd.DataFrame(results)


OUTPUT.parent.mkdir(
    exist_ok=True
)


df.to_csv(
    OUTPUT,
    index=False
)


print("\nFINAL COMPARISON")
print(df)


print("\nSaved:")
print(OUTPUT)


print("\nDONE")