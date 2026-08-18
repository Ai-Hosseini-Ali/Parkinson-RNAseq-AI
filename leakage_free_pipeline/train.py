import os
import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer
)

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report
)



print("==============================")
print("FINAL LEAKAGE FREE PARKINSON MODEL")
print("==============================")



# =====================================
# Custom functions
# =====================================


def log_transform(X):
    """
    Log2 normalization
    """
    return np.log2(X + 1)




# =====================================
# Paths
# =====================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "GSE99039_final_dataset.csv"
)



MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



print("\nProject:")
print(BASE_DIR)


print("\nDataset:")
print(DATA_PATH)




# =====================================
# Load dataset
# =====================================


df = pd.read_csv(
    DATA_PATH
)



print("\nOriginal dataset:")
print(df.shape)



print("\nColumns:")
print(df.columns[:10])




# =====================================
# Remove identifiers
# =====================================


for col in [
    "Unnamed: 0",
    "Sample"
]:

    if col in df.columns:

        df = df.drop(
            columns=[col]
        )





# =====================================
# Prepare X y
# =====================================


y = df["Disease"].map(
    {
        "CONTROL":0,
        "IPD":1
    }
)



X = df.drop(
    columns=["Disease"]
)



# keep only genes

X = X.select_dtypes(
    include=[np.number]
)



print("\nFeature matrix:")
print(X.shape)



print("\nClasses:")
print(y.value_counts())



gene_names = X.columns.to_list()




# =====================================
# Leakage free Pipeline
# =====================================


pipeline = Pipeline([


    (
        "log_transform",

        FunctionTransformer(
            log_transform
        )

    ),



    (
        "feature_selection",

        SelectKBest(
            score_func=f_classif,
            k=500
        )

    ),



    (
        "scaler",

        StandardScaler()

    ),



    (
        "classifier",

        LogisticRegression(

            max_iter=5000,

            random_state=42

        )

    )

])





# =====================================
# Cross Validation
# =====================================


cv = StratifiedKFold(

    n_splits=10,

    shuffle=True,

    random_state=42

)



print("\nRunning cross validation...")



prob = cross_val_predict(

    pipeline,

    X,

    y,

    cv=cv,

    method="predict_proba"

)[:,1]




pred = (

    prob >= 0.5

).astype(int)





auc = roc_auc_score(

    y,

    prob

)



acc = accuracy_score(

    y,

    pred

)




print("\n===================")
print("RESULTS")
print("===================")



print(
    "AUC:",
    auc
)



print(
    "Accuracy:",
    acc
)



print(
    classification_report(
        y,
        pred
    )
)




# =====================================
# Train final model
# =====================================


print("\nTraining final pipeline...")



pipeline.fit(

    X,

    y

)




# =====================================
# Extract selected genes
# =====================================


selector = pipeline.named_steps[
    "feature_selection"
]



mask = selector.get_support()



selected_genes = np.array(
    gene_names
)[mask]



print("\nSelected genes:")
print(len(selected_genes))


print(
    selected_genes[:20]
)




# =====================================
# Save pipeline
# =====================================


pipeline_file = os.path.join(

    MODEL_DIR,

    "final_leakage_free_pipeline.pkl"

)



joblib.dump(

    pipeline,

    pipeline_file

)




# =====================================
# Save selected genes
# =====================================


genes_file = os.path.join(

    MODEL_DIR,

    "final_selected_500_genes.txt"

)



with open(

    genes_file,

    "w"

) as f:


    for gene in selected_genes:

        f.write(
            gene + "\n"
        )





# =====================================
# Save metadata
# =====================================


metadata = {


    "dataset":

    "GSE99039",



    "samples":

    int(X.shape[0]),



    "original_features":

    int(len(gene_names)),



    "selected_features":

    int(len(selected_genes)),



    "feature_selection":

    "SelectKBest f_classif inside pipeline",



    "normalization":

    "log2(X+1)",



    "scaler":

    "StandardScaler inside pipeline",



    "model":

    "LogisticRegression"

}



metadata_file = os.path.join(

    MODEL_DIR,

    "model_metadata.pkl"

)



joblib.dump(

    metadata,

    metadata_file

)




print("\nSaved:")

print(
    pipeline_file
)


print(
    genes_file
)


print(
    metadata_file
)



print("\nFINISHED")