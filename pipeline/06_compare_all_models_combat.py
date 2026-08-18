import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

import joblib


print("="*60)
print("COMPARE ALL MODELS - COMBAT 14 GENE SIGNATURE")
print("="*60)


# Load harmonized data

data = pd.read_csv(
    "data/combat_harmonized_14genes.csv",
    index_col=0
)


print("Dataset:", data.shape)


# labels
labels = []

for name in data.index:

    if "PD" in name:
        labels.append(1)

    elif "CC" in name:
        labels.append(0)

    else:
        labels.append(np.nan)


data["Label"] = labels

data = data.dropna()


X = data.drop(
    columns=["Label"]
)

y = data["Label"]


print("\nClasses:")
print(y.value_counts())


# Models

models = {


"Logistic Regression":

Pipeline([
    ("scaler",StandardScaler()),
    ("model",
     LogisticRegression(
         max_iter=2000
     ))
]),



"SVM":

Pipeline([
    ("scaler",StandardScaler()),
    ("model",
     SVC(
         probability=True,
         kernel="rbf"
     ))
]),



"Random Forest":

RandomForestClassifier(
    n_estimators=500,
    random_state=42
),



"Gradient Boosting":

GradientBoostingClassifier(
    random_state=42
),



"MLP":

Pipeline([
    ("scaler",StandardScaler()),
    ("model",
     MLPClassifier(
         hidden_layer_sizes=(32,16),
         max_iter=2000,
         random_state=42
     ))
]),



"KNN":

Pipeline([
    ("scaler",StandardScaler()),
    ("model",
     KNeighborsClassifier(
         n_neighbors=5
     ))
])

}



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



results=[]


for name,model in models.items():

    print("\nRunning:",name)

    auc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )


    acc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )


    results.append({

        "Model":name,

        "AUC_mean":
        auc.mean(),

        "AUC_std":
        auc.std(),

        "Accuracy_mean":
        acc.mean(),

        "Accuracy_std":
        acc.std()

    })


    print(
        "AUC:",
        round(auc.mean(),3),
        "+/-",
        round(auc.std(),3)
    )

    print(
        "Accuracy:",
        round(acc.mean(),3)
    )



results=pd.DataFrame(results)

results=results.sort_values(
    "AUC_mean",
    ascending=False
)


print("\nFINAL RESULTS")
print(results)


results.to_csv(
    "results/combat_all_models_comparison.csv",
    index=False
)


print("\nSaved:")
print(
"results/combat_all_models_comparison.csv"
)