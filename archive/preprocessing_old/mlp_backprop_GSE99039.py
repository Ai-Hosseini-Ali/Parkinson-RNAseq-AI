import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# Load dataset
# =========================

df = pd.read_csv(
    "data/GSE99039_top100_dataset.csv"
)


print("Dataset:")
print(df.shape)


# =========================
# Features / Labels
# =========================

X = df.drop(columns=["Disease"])

y = df["Disease"]


print("\nFeatures:")
print(X.shape)

print("Labels:")
print(y.shape)


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Standardization
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


print("\nScaling done")


# =========================
# MLP Model
# Backpropagation
# =========================


model = MLPClassifier(

    hidden_layer_sizes=(64,32),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=500,

    random_state=42,

    early_stopping=True

)



# =========================
# Training
# =========================

model.fit(
    X_train,
    y_train
)


print("\nTraining finished")


# =========================
# Prediction
# =========================

y_pred = model.predict(
    X_test
)



# =========================
# Evaluation
# =========================

print("\n======================")
print("MLP Backpropagation")
print("======================")

print(
    "Accuracy:",
    accuracy_score(y_test,y_pred)
)


print(
    "Precision:",
    precision_score(
        y_test,
        y_pred,
        pos_label="IPD"
    )
)


print(
    "Recall:",
    recall_score(
        y_test,
        y_pred,
        pos_label="IPD"
    )
)


print(
    "F1:",
    f1_score(
        y_test,
        y_pred,
        pos_label="IPD"
    )
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================
# Cross Validation
# =========================

scores = cross_val_score(
    model,
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring="accuracy"
)


print("\n5 Fold CV:")

print(scores)

print(
    "Mean CV Accuracy:",
    scores.mean()
)