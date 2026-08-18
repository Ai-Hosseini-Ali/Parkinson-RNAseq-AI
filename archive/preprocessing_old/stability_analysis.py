import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


# ==========================
# Load candidate genes
# ==========================

candidate = pd.read_csv(
    "data/pathway_input_biomarkers.csv"
)

candidate_genes = candidate["Gene"].tolist()


print("Candidate genes:")
print(len(candidate_genes))



# ==========================
# Load expression
# ==========================

expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


expression = expression.rename(
    columns={
        "Unnamed: 0": "Sample"
    }
)


labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)



data = expression.merge(
    labels[["Sample","Label"]],
    on="Sample"
)



print("\nDataset:")
print(data.shape)



# فقط ژن‌هایی که وجود دارند

genes_available = [
    g for g in candidate_genes
    if g in data.columns
]


print("\nAvailable candidate genes:")
print(len(genes_available))



# ==========================
# Stability
# ==========================


iterations = 100


selection_count = {
    gene:0
    for gene in genes_available
}



sss = StratifiedShuffleSplit(
    n_splits=iterations,
    test_size=0.2,
    random_state=42
)



for i,(train_idx,test_idx) in enumerate(
    sss.split(
        data,
        data["Label"]
    )
):

    train = data.iloc[train_idx]


    scores = {}


    for gene in genes_available:

        corr = abs(
            train[[gene,"Label"]]
            .corr()
            .iloc[0,1]
        )

        scores[gene]=corr



    selected = (
        pd.Series(scores)
        .sort_values(
            ascending=False
        )
        .head(15)
        .index
    )


    for gene in selected:
        selection_count[gene]+=1



    if (i+1)%10==0:
        print(
            "Completed:",
            i+1,
            "/",
            iterations
        )



# ==========================
# Results
# ==========================


result = pd.DataFrame(
    {
        "Gene":selection_count.keys(),
        "Selected_times":selection_count.values()
    }
)


result["Stability_%"] = (
    result["Selected_times"]
    /
    iterations
    *
    100
)


result = result.sort_values(
    "Stability_%",
    ascending=False
)



print("\n====================")
print("STABILITY RESULTS")
print("====================")

print(result.head(30))



result.to_csv(
    "data/gene_stability_results.csv",
    index=False
)


print("\nSaved:")
print(
    "data/gene_stability_results.csv"
)