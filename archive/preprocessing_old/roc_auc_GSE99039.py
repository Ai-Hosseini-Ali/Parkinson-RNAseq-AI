import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, auc

# =====================================
# Load dataset
# =====================================

df = pd.read_csv("data/GSE99039_top100_dataset.csv")

print(df.shape)

X = df.drop(columns=["Disease"])
y = df["Disease"]

# تبدیل برچسب‌ها به صفر و یک
y = y.map({
    "CONTROL": 0,
    "IPD": 1
})

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# Standardization
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# Models
# =====================================

models = {

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(64,32),
        max_iter=1000,
        random_state=42
    )

}

# =====================================
# ROC Curves
# =====================================

plt.figure(figsize=(8,7))

for name, model in models.items():

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    roc_auc = auc(fpr, tpr)

    print(name, "AUC =", roc_auc)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

# Random line

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves (GSE99039)")

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/GSE99039_ROC.png",
    dpi=300
)

plt.show()

print("\nROC figure saved successfully.")