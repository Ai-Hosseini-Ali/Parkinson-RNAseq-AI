import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score


# ==========================
# Load ranking components
# ==========================

ml = pd.read_csv(
    "data/GSE99039_feature_importance.csv"
)

de = pd.read_csv(
    "data/GSE99039_DE_FDR.csv"
)

ppi = pd.read_csv(
    "data/string_hub_genes.csv"
)



# normalize columns

ml = ml.rename(
    columns={
        "Importance":"ML_score"
    }
)


de = de.rename(
    columns={
        "Gene":"Gene"
    }
)



ppi = ppi.rename(
    columns={
        "Gene":"Gene"
    }
)



# ==========================
# Prepare scores
# ==========================

ml["ML_score"] = (
    ml["ML_score"] - ml["ML_score"].min()
) / (
    ml["ML_score"].max()
    -
    ml["ML_score"].min()
)



de["DE_score"] = 1 - de["FDR"]



de["DE_score"] = (
    de["DE_score"] - de["DE_score"].min()
) / (
    de["DE_score"].max()
    -
    de["DE_score"].min()
)



ppi["PPI_score"] = (
    ppi["Degree"] - ppi["Degree"].min()
) / (
    ppi["Degree"].max()
    -
    ppi["Degree"].min()
)



# merge

rank = (
    ml[["Gene","ML_score"]]
    .merge(
        de[["Gene","DE_score"]],
        on="Gene",
        how="outer"
    )
    .merge(
        ppi[["Gene","PPI_score"]],
        on="Gene",
        how="outer"
    )
)



rank = rank.fillna(0)



# ==========================
# Weight combinations
# ==========================

weights = {

"A_current":
(0.5,0.3,0.2),

"B":
(0.4,0.3,0.3),

"C":
(0.3,0.3,0.4),

"D":
(0.3,0.2,0.5)

}



# Load expression

expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


expression = expression.rename(
    columns={
        "Unnamed: 0":"Sample"
    }
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


data = expression.merge(
    labels[["Sample","Label"]],
    on="Sample"
)



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



results=[]



for name,w in weights.items():

    print("\nTesting",name)


    temp = rank.copy()


    temp["Final"] = (
        w[0]*temp["ML_score"]
        +
        w[1]*temp["DE_score"]
        +
        w[2]*temp["PPI_score"]
    )


    genes = (
        temp.sort_values(
            "Final",
            ascending=False
        )
        .head(15)["Gene"]
        .tolist()
    )


    genes = [
        g for g in genes
        if g in data.columns
    ]


    print(genes)



    X=data[genes]

    y=data["Label"]


    X=StandardScaler().fit_transform(X)



    model=SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )


    prob=cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:,1]


    auc=roc_auc_score(
        y,
        prob
    )


    acc=accuracy_score(
        y,
        prob>=0.5
    )


    results.append(
        {
            "Method":name,
            "AUC":auc,
            "Accuracy":acc,
            "Genes":",".join(genes)
        }
    )



results=pd.DataFrame(results)


print("\nRESULTS")
print(results[["Method","AUC","Accuracy"]])



results.to_csv(
    "data/network_weight_results.csv",
    index=False
)


print("\nSaved")