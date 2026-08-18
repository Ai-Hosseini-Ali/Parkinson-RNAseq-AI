import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score


print("="*40)
print(" FINAL PARKINSON MODEL BENCHMARK ")
print("="*40)


# ==============================
# Locked gene signature
# ==============================

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


print("\nLocked genes:")
print(genes)


# ==============================
# Paths
# ==============================

expression_file = r"C:\Users\alihoseini\Desktop\Parkinson_AI\data\GSE99039_final_dataset.csv"

label_file = r"C:\Users\alihoseini\Desktop\Parkinson_AI\data\GSE99039_labels.csv"


print("\nExpression file:")
print(expression_file)

print("\nLabel file:")
print(label_file)



if not os.path.exists(expression_file):
    raise FileNotFoundError(expression_file)

if not os.path.exists(label_file):
    raise FileNotFoundError(label_file)



# ==============================
# Load data
# ==============================

print("\nLoading data...")


X = pd.read_csv(expression_file, index_col=0)

y = pd.read_csv(label_file)



print("\nExpression shape:")
print(X.shape)

print("\nLabel columns:")
print(y.columns)



# ==============================
# Fix labels
# ==============================

if "Label" in y.columns:
    y = y["Label"]

else:
    y = y.iloc[:,0]


print("\nLabels:")
print(y.value_counts())



# ==============================
# Select genes
# ==============================

available = []

for g in genes:
    if g in X.columns:
        available.append(g)


print("\nAvailable genes:")
print(available)


missing = set(genes)-set(available)

print("\nMissing genes:")
print(missing)



X = X[available]


# transpose if needed

if X.shape[0] != len(y):

    print("\nTransposing expression matrix...")
    X = X.T



print("\nFinal X:")
print(X.shape)

print("Final y:")
print(y.shape)



# ==============================
# Split
# ==============================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ==============================
# Scaling
# ==============================

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# ==============================
# Models
# ==============================

models = {

"Logistic Regression":
LogisticRegression(
    max_iter=5000
),

"SVM":
SVC(
    probability=True
),

"Random Forest":
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

}



results=[]



for name,model in models.items():

    print("\nTraining:",name)

    model.fit(
        X_train,
        y_train
    )


    pred=model.predict(
        X_test
    )


    prob=model.predict_proba(
        X_test
    )[:,1]


    acc=accuracy_score(
        y_test,
        pred
    )


    auc=roc_auc_score(
        y_test,
        prob
    )


    print("Accuracy:",acc)
    print("AUC:",auc)


    results.append(
        [
            name,
            acc,
            auc
        ]
    )



# ==============================
# Save results
# ==============================

result_df=pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC"
    ]
)


os.makedirs(
    "results",
    exist_ok=True
)


result_df.to_csv(
    "results/final_model_benchmark.csv",
    index=False
)


print("\nFINAL RESULTS")
print(result_df)


print("\nSaved:")
print("results/final_model_benchmark.csv")