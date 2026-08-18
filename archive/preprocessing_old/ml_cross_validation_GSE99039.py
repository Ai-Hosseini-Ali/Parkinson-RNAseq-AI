import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/GSE99039_top100_dataset.csv"
)


print("Dataset:")
print(df.shape)


X = df.drop(columns=["Disease"])
y = df["Disease"]



# =========================
# Models
# =========================


models = {

"SVM":
Pipeline([
    ("scale", StandardScaler()),
    ("model", SVC(kernel="rbf"))
]),


"Logistic Regression":
Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
]),


"Random Forest":
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

}



# =========================
# Cross Validation
# =========================


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for name, model in models.items():

    print("\n================")
    print(name)
    print("================")


    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted"
        ]
    )


    print(
        "Accuracy:",
        np.mean(results["test_accuracy"])
    )

    print(
        "Precision:",
        np.mean(results["test_precision_weighted"])
    )

    print(
        "Recall:",
        np.mean(results["test_recall_weighted"])
    )

    print(
        "F1:",
        np.mean(results["test_f1_weighted"])
    )