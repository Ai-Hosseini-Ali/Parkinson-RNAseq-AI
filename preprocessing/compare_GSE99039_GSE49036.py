import pandas as pd
import numpy as np
import os


print("""
==========================================
COMPARE DATASETS
GSE99039 vs GSE49036
8 GENE SIGNATURE
==========================================
""")


# ==========================
# Paths
# ==========================

TRAIN = "data/GSE99039_top100_dataset.csv"
VALID = "data/GSE49036_validation_8gene.csv"

OUTPUT = "results/gene_distribution_comparison.csv"


GENES = [
    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"
]


# ==========================
# Load datasets
# ==========================

print("Loading datasets...")


train = pd.read_csv(
    TRAIN
)

valid = pd.read_csv(
    VALID
)


print("\nTraining:")
print(train.shape)


print("\nValidation:")
print(valid.shape)



# ==========================
# Select genes
# ==========================

train_gene = train[GENES]

valid_gene = valid[GENES]



# ==========================
# Statistics
# ==========================

results=[]


for g in GENES:

    results.append({

        "Gene":g,

        "GSE99039_mean":
            train_gene[g].mean(),

        "GSE99039_std":
            train_gene[g].std(),

        "GSE49036_mean":
            valid_gene[g].mean(),

        "GSE49036_std":
            valid_gene[g].std(),

        "Mean_difference":
            valid_gene[g].mean()
            -
            train_gene[g].mean()
    })



result = pd.DataFrame(results)



print("\n================================")
print("Gene distribution")
print("================================")


print(result)



# ==========================
# Save
# ==========================

os.makedirs(
    "results",
    exist_ok=True
)


result.to_csv(
    OUTPUT,
    index=False
)



print("""
==========================================

Finished

Saved:
results/gene_distribution_comparison.csv

==========================================
""")