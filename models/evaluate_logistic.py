import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# =========================
# Load data
# =========================

df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)

genes = df["Geneid"]
counts = df.drop("Geneid", axis=1)

selected = pd.read_csv("data/top500_genes.csv")

top_genes = selected["Gene"].tolist()

filtered = counts[genes.isin(top_genes)]

X = filtered.T

y = []

for sample in X.index:

    if "PD" in sample:
        y.append(1)
    else:
        y.append(0)

y = np.array(y)

# =========================
# Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    stratify=y,

    random_state=42
)

# =========================
# Model
# =========================

model = Pipeline([

    ("scaler", StandardScaler()),

    ("logistic",
        LogisticRegression(max_iter=5000)
    )

])

model.fit(X_train, y_train)

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]

# =========================
# Metrics
# =========================

print("Accuracy :", accuracy_score(y_test,pred))

print("Precision:", precision_score(y_test,pred))

print("Recall   :", recall_score(y_test,pred))

print("F1-score :", f1_score(y_test,pred))

auc = roc_auc_score(y_test,prob)

print("AUC      :",auc)

print("\nConfusion Matrix")

print(confusion_matrix(y_test,pred))

# =========================
# ROC
# =========================

fpr,tpr,_ = roc_curve(y_test,prob)

plt.figure(figsize=(6,6))

plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Logistic Regression")

plt.legend()

plt.tight_layout()

plt.savefig("figures/Figure5_ROC_Logistic.png",dpi=300)

plt.show()