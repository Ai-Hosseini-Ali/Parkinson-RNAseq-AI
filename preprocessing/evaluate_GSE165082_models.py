import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


print("="*60)
print("Loading GSE165082 Dataset")
print("="*60)


# ===============================
# Load expression data
# ===============================

df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)


# Gene column
genes = df["Geneid"]


X = df.drop(
    "Geneid",
    axis=1
)


# ===============================
# Filter low expression genes
# ===============================

X = X[X.sum(axis=1) >= 10]


# ===============================
# Normalize
# ===============================

X = np.log2(
    X + 1
)


# ===============================
# Transpose
# samples as rows
# ===============================

X = X.T


# ===============================
# Create labels
# ===============================

labels = []


for sample in X.index:

    if "PD" in sample:
        labels.append(1)

    elif "CC" in sample:
        labels.append(0)

    else:
        labels.append(None)



y = pd.Series(
    labels,
    index=X.index
)


# remove unknown samples

mask = y.notna()

X = X[mask]

y = y[mask]


print("Samples:")
print(X.shape)


print("\nClasses:")
print(y.value_counts())



# ===============================
# Select top 500 genes
# ===============================


correlations = []


for gene in X.columns:

    corr = abs(
        np.corrcoef(
            X[gene],
            y
        )[0,1]
    )

    correlations.append(
        corr
    )


gene_scores = pd.DataFrame({

    "Gene":X.columns,

    "Score":correlations

})


top_genes = gene_scores.sort_values(
    "Score",
    ascending=False
).head(500)["Gene"]



X = X[top_genes]



print("\nSelected genes:")
print(X.shape)



# ===============================
# Split
# ===============================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42

)



# ===============================
# Scaling
# ===============================


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



# ===============================
# Models
# ===============================


models = {


"Logistic Regression":
LogisticRegression(
    max_iter=2000
),


"SVM":
SVC(
    kernel="linear",
    probability=True
),


"MLP":
MLPClassifier(
    hidden_layer_sizes=(32,16),
    max_iter=2000,
    random_state=42
)

}



results=[]



for name,model in models.items():


    print("\n")
    print(name)


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

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )


    results.append({

        "Dataset":"GSE165082",

        "Model":name,

        "Number_of_Genes":500,

        "Accuracy":round(acc,4),

        "AUC":round(auc,4)

    })



# Save

results=pd.DataFrame(results)


print("\nFINAL TABLE")
print(results)



results.to_csv(
    "results/GSE165082_model_comparison.csv",
    index=False
)


print("\nSaved")