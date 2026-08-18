import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("data/GSE99039_top100_dataset.csv")

print("Dataset shape:")
print(df.shape)

# ==========================================
# Features and labels
# ==========================================

X = df.drop(columns=["Disease"])
y = df["Disease"]

print("\nFeatures:", X.shape)
print("Labels:", y.shape)

# ==========================================
# Train / Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)

# ==========================================
# Random Forest model
# ==========================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = rf.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, pos_label="IPD")
rec = recall_score(y_test, y_pred, pos_label="IPD")
f1 = f1_score(y_test, y_pred, pos_label="IPD")

print("\n==============================")
print("Random Forest Results")
print("==============================")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))