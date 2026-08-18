import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


print("================================")
print("External Validation Batch Correction")
print("================================")


# ===============================
# Final signature genes
# ===============================

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


# ===============================
# Load GSE99039
# ===============================

print("\nLoading GSE99039...")


train = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


print("Expression:")
print(train.shape)


# Fix sample column

if "Sample" not in train.columns:

    if "Unnamed: 0" in train.columns:

        train.rename(
            columns={
                "Unnamed: 0":"Sample"
            },
            inplace=True
        )


print(train.head())


train = train.merge(
    labels,
    on="Sample"
)


print("\nTraining merged:")
print(train.shape)


# ===============================
# Load external GSE165082
# ===============================

print("\nLoading GSE165082...")


ext = pd.read_csv(
    "data/GSE165082_gene_symbol.csv"
)


ext = ext.set_index(
    "GeneSymbol"
)


# genes as columns

ext = ext.T


print("External:")
print(ext.shape)



# Labels external

ext_labels=[]


for s in ext.index:

    if "PD" in s:
        ext_labels.append(1)

    elif "CC" in s:
        ext_labels.append(0)


y_ext=np.array(ext_labels)



print("\nExternal classes:")
print(
pd.Series(y_ext).value_counts()
)



# ===============================
# Common genes
# ===============================


available = [
    g for g in genes
    if g in train.columns and g in ext.columns
]


print("\nAvailable genes:")
print(available)

print(
"Number:",
len(available)
)


X_train = train[available]

y_train = train["Label"]


X_ext = ext[available]



# ===============================
# Combine
# ===============================


combined = pd.concat(
    [
        X_train,
        X_ext
    ],
    axis=0
)



# log transform

combined = np.log2(
    combined + 1
)



batch = np.array(
    [0]*len(X_train)
    +
    [1]*len(X_ext)
)



# ===============================
# Batch centering
# ===============================


corrected = combined.copy()


global_mean = corrected.mean()


for b in np.unique(batch):

    idx = batch == b

    corrected.iloc[idx] = (
        corrected.iloc[idx]
        -
        corrected.iloc[idx].mean()
        +
        global_mean
    )



X_train_corr = corrected.iloc[
    :len(X_train)
]


X_ext_corr = corrected.iloc[
    len(X_train):
]



# ===============================
# Scaling
# ===============================


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_corr
)


X_ext_scaled = scaler.transform(
    X_ext_corr
)



# ===============================
# Models
# ===============================


models = {

"Logistic":
LogisticRegression(
    max_iter=5000,
    random_state=42
),


"SVM":
SVC(
    probability=True,
    random_state=42
)

}



results=[]


for name,model in models.items():


    print("\n================")
    print(name)
    print("================")


    model.fit(
        X_train_scaled,
        y_train
    )


    prob = model.predict_proba(
        X_ext_scaled
    )[:,1]


    pred = model.predict(
        X_ext_scaled
    )


    auc = roc_auc_score(
        y_ext,
        prob
    )


    acc = accuracy_score(
        y_ext,
        pred
    )


    print(
        "AUC:",
        auc
    )


    print(
        "Accuracy:",
        acc
    )


    print(
        confusion_matrix(
            y_ext,
            pred
        )
    )


    results.append(
        [
            name,
            auc,
            acc
        ]
    )



# ===============================
# Save results
# ===============================


pd.DataFrame(
    results,
    columns=[
        "Model",
        "AUC",
        "Accuracy"
    ]
).to_csv(
    "data/batch_external_results.csv",
    index=False
)



# ===============================
# PCA
# ===============================


pca=PCA(
    n_components=2
)


pc=pca.fit_transform(
    corrected
)



plt.figure(
    figsize=(7,5)
)


plt.scatter(
    pc[:len(X_train),0],
    pc[:len(X_train),1],
    label="GSE99039"
)


plt.scatter(
    pc[len(X_train):,0],
    pc[len(X_train):,1],
    label="GSE165082"
)


plt.legend()

plt.title(
    "Batch Corrected PCA"
)


plt.savefig(
    "data/batch_corrected_PCA.png"
)


print("\nFinished")
print(
"Saved:"
)

print(
"data/batch_external_results.csv"
)

print(
"data/batch_corrected_PCA.png"
)