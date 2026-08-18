import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# =====================================================
# 1. Load Dataset
# =====================================================

print("="*60)
print("Loading GSE99039 Dataset")
print("="*60)


expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


# اصلاح نام ستون Sample
expression = expression.rename(
    columns={
        "Unnamed: 0": "Sample"
    }
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


# Merge expression + labels

data = expression.merge(
    labels[["Sample", "Label"]],
    on="Sample"
)


print("Dataset shape:")
print(data.shape)



# =====================================================
# 2. Final Gene Signature
# =====================================================


genes = [

    "PTGDS",
    "KIR2DL1",
    "KIR2DL3",
    "LILRB1",
    "KIAA0319L",
    "PPP4C",
    "TYROBP",
    "MBOAT7",
    "MMP9",
    "HLA-C",
    "LRRC25",
    "KIR3DL1",
    "BCL2",
    "RHOG"

]


# بررسی وجود ژن‌ها

missing = [
    g for g in genes
    if g not in data.columns
]


if len(missing) > 0:

    print("Missing genes:")
    print(missing)

    exit()



X = data[genes]

y = data["Label"]



print("\nClasses:")
print(y.value_counts())



# =====================================================
# 3. Train Test Split
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42

)



print("\nTrain:")
print(X_train.shape)


print("Test:")
print(X_test.shape)



# =====================================================
# 4. Scaling
# =====================================================


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



# =====================================================
# 5. Models
# =====================================================


models = {


"Logistic Regression":

LogisticRegression(
    max_iter=2000,
    random_state=42
),



"SVM":

SVC(

    kernel="linear",

    probability=True,

    random_state=42

),



"MLP":

MLPClassifier(

    hidden_layer_sizes=(32,16),

    max_iter=2000,

    random_state=42

)

}



# =====================================================
# 6. Evaluation
# =====================================================


results = []



for name, model in models.items():


    print("\n")
    print("="*40)
    print(name)
    print("="*40)


    model.fit(

        X_train_scaled,

        y_train

    )


    prediction = model.predict(

        X_test_scaled

    )


    probability = model.predict_proba(

        X_test_scaled

    )[:,1]



    accuracy = accuracy_score(

        y_test,

        prediction

    )


    auc = roc_auc_score(

        y_test,

        probability

    )



    cm = confusion_matrix(

        y_test,

        prediction

    )



    print(
        "Accuracy:",
        round(accuracy,4)
    )


    print(
        "AUC:",
        round(auc,4)
    )


    print(
        "Confusion Matrix:"
    )

    print(cm)



    results.append({

        "Dataset":"GSE99039",

        "Model":name,

        "Number_of_Genes":len(genes),

        "Accuracy":round(accuracy,4),

        "AUC":round(auc,4)

    })




# =====================================================
# 7. Save Final Table
# =====================================================


results = pd.DataFrame(results)


print("\n")
print("="*60)
print("FINAL MODEL COMPARISON TABLE")
print("="*60)

print(results)



results.to_csv(

    "results/model_comparison_final.csv",

    index=False

)



print("\nSaved:")
print(
    "results/model_comparison_final.csv"
)