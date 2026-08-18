import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# Load dataset
# ==========================

df = pd.read_csv("data/GSE99039_top100_dataset.csv")

print("Dataset:")
print(df.shape)

X = df.drop(columns=["Disease"])
y = df["Disease"]

# ==========================
# Train/Test
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Standardization
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# SVM
# ==========================

model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Results
# ==========================

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nPrecision")
print(precision_score(y_test, y_pred, pos_label="IPD"))

print("\nRecall")
print(recall_score(y_test, y_pred, pos_label="IPD"))

print("\nF1")
print(f1_score(y_test, y_pred, pos_label="IPD"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))