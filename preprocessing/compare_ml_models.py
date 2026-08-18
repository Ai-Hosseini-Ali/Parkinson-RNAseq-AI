import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)


# ==========================
# Load data
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
# Biomarkers
# ==========================

biomarkers = pd.read_csv(
    "data/final_ranked_biomarkers.csv"
)


genes = biomarkers.head(20)["Gene"].tolist()


genes = [
    g for g in genes
    if g in data.columns
]


print("\nGenes:")
print(genes)



X = data[genes]

y = data["Label"]



# Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)



# ==========================
# Models
# ==========================

models = {

"Logistic Regression":
LogisticRegression(
    max_iter=2000,
    random_state=42
),


"SVM":
SVC(
    probability=True,
    kernel="rbf",
    random_state=42
),


"Random Forest":
RandomForestClassifier(
    n_estimators=500,
    random_state=42
),


"Gradient Boosting":
GradientBoostingClassifier(
    random_state=42
)

}



# ==========================
# CV
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



results = []



for name, model in models.items():

    print("\nRunning:", name)


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
            "Model": name,
            "AUC": auc,
            "Accuracy": acc
        }
    )



# ==========================
# Results
# ==========================

results = pd.DataFrame(results)

results = results.sort_values(
    "AUC",
    ascending=False
)


print("\n================")
print(results)
print("================")



results.to_csv(
    "data/model_comparison_results.csv",
    index=False
)


print("\nSaved:")
print("data/model_comparison_results.csv")