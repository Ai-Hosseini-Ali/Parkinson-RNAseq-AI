import pandas as pd
import numpy as np
from pathlib import Path


print("="*60)
print(" CROSS PLATFORM GENE EXPRESSION DIAGNOSTIC ")
print(" GSE99039 vs GSE165082")
print("="*60)


# ==============================
# Paths
# ==============================

TRAIN_FILE = Path(
    "data/GSE99039_final_dataset.csv"
)

EXTERNAL_FILE = Path(
    "data/GSE165082_normalized_log_cpm.csv"
)

GENE_FILE = Path(
    "model/final_genes.txt"
)

OUTPUT = Path(
    "results/cross_platform_gene_diagnostic.csv"
)


# ==============================
# Load genes
# ==============================

genes = [
    x.strip()
    for x in open(GENE_FILE)
    if x.strip()
]

print("\nLocked genes:")
print(genes)


# ==============================
# Load datasets
# ==============================

print("\nLoading training data...")

train = pd.read_csv(TRAIN_FILE)

print("Training:")
print(train.shape)


print("\nLoading external data...")

external = pd.read_csv(EXTERNAL_FILE)

print("External:")
print(external.shape)



# ==============================
# Detect gene orientation
# ==============================

def prepare_gene_matrix(df, genes):

    cols = set(df.columns)

    if all(g in cols for g in genes):

        print("Genes detected in columns")

        X = df[genes].copy()

    elif "Gene" in df.columns:

        print("Genes detected in rows")

        X = (
            df
            .set_index("Gene")
            .loc[genes]
            .T
        )

    else:

        raise ValueError(
            "Gene format not recognized"
        )

    return X



train_X = prepare_gene_matrix(
    train,
    genes
)


external_X = prepare_gene_matrix(
    external,
    genes
)



# ==============================
# Diagnostic statistics
# ==============================


results=[]


for gene in genes:

    train_values = train_X[gene].astype(float)

    ext_values = external_X[gene].astype(float)


    train_mean = train_values.mean()
    ext_mean = ext_values.mean()


    train_std = train_values.std()
    ext_std = ext_values.std()


    pooled_std = np.sqrt(
        (
            train_std**2 +
            ext_std**2
        ) / 2
    )


    if pooled_std == 0:

        effect = 0

    else:

        effect = (
            ext_mean - train_mean
        ) / pooled_std



    results.append({

        "Gene":gene,

        "Train_mean":train_mean,

        "External_mean":ext_mean,

        "Mean_difference":
            ext_mean-train_mean,

        "Train_std":
            train_std,

        "External_std":
            ext_std,

        "Standardized_shift":
            effect
    })



result_df = pd.DataFrame(results)


# sort by strongest shift

result_df["Absolute_shift"] = (
    result_df["Standardized_shift"]
    .abs()
)


result_df = result_df.sort_values(
    "Absolute_shift",
    ascending=False
)



# ==============================
# Save
# ==============================

OUTPUT.parent.mkdir(
    exist_ok=True
)


result_df.to_csv(
    OUTPUT,
    index=False
)


print("\n")
print("="*60)
print("RESULT")
print("="*60)

print(result_df)


print("\nSaved:")
print(OUTPUT)

print("\nDONE")