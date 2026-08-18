import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, accuracy_score


# ==========================
# Load expression
# ==========================

expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)

expression = expression.rename(
    columns={
        "Unnamed: 0": "Sample"
    }
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


data = expression.merge(
    labels[["Sample","Label"]],
    on="Sample"
)


print("Dataset:")
print(data.shape)



# ==========================
# Ranked genes
# ==========================

ranked = pd.read_csv(
    "data/final_ranked_biomarkers.csv"
)


all_genes = ranked["Gene"].tolist()



# ==========================
# Test sizes
# ==========================

sizes = [
    5,
    10,
    15,
    20,
    30,
    50,
    100
]


results = []



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



for size in sizes:

    genes = [
        g for g in all_genes[:size]
        if g in data.columns
    ]


    print("\nTesting:")
    print(size, "genes")



    X = data[genes]

    y = data["Label"]



    scaler = StandardScaler()

    X = scaler.fit_transform(X)



    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )



    prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:,1]



    pred = (
        prob >= 0.5
    ).astype(int)



    auc = roc_auc_score(
        y,
        prob
    )


    acc = accuracy_score(
        y,
        pred
    )



    results.append(
        {
            "Gene_Number": size,
            "Genes": ",".join(genes),
            "AUC": auc,
            "Accuracy": acc
        }
    )



results = pd.DataFrame(results)


results = results.sort_values(
    "AUC",
    ascending=False
)


print("\n===================")
print(results[[
    "Gene_Number",
    "AUC",
    "Accuracy"
]])
print("===================")



results.to_csv(
    "data/signature_size_optimization.csv",
    index=False
)


print("\nSaved:")
print(
    "data/signature_size_optimization.csv"
)
