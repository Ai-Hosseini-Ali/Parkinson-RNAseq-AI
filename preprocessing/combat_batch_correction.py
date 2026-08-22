import pandas as pd
import numpy as np
import os

from neuroCombat import neuroCombat


print("""
==========================================
COMBAT BATCH CORRECTION
GSE99039 + GSE49036
8 GENE SIGNATURE
==========================================
""")


TRAIN_DATA = "data/GSE99039_8gene_labeled.csv"
VALID_DATA = "data/GSE49036_validation_8gene.csv"

OUTPUT = "data/combat_corrected_8gene.csv"


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
# Load
# ==========================

print("Loading datasets...")


train = pd.read_csv(
    TRAIN_DATA,
    index_col=0
)


valid = pd.read_csv(
    VALID_DATA
)


print("\nTraining raw:")
print(train.shape)


print("\nValidation raw:")
print(valid.shape)



# ==========================
# Prepare validation
# ==========================

valid = valid.set_index("Sample")



# ==========================
# Select genes
# ==========================

print("\nChecking genes...")


for name, df in [("Training", train), ("Validation", valid)]:

    missing = [
        g for g in GENES
        if g not in df.columns
    ]

    if missing:
        raise Exception(
            f"{name} missing genes: {missing}"
        )

    print(
        f"{name}: all genes found"
    )



train_X = train[GENES]

valid_X = valid[GENES]



print("\nTraining selected:")
print(train_X.shape)


print("Validation selected:")
print(valid_X.shape)



# ==========================
# Combine for ComBat
# ==========================

# ComBat format:
# rows = genes
# columns = samples


combined = pd.concat(
    [
        train_X.T,
        valid_X.T
    ],
    axis=1
)


print("\nCombined matrix:")
print(combined.shape)



# ==========================
# Batch information
# ==========================

batch = np.array(
    [0] * train_X.shape[0]
    +
    [1] * valid_X.shape[0]
)


print("\nBatch distribution:")
print(
    pd.Series(batch).value_counts()
)



# ==========================
# Run ComBat
# ==========================

print("""
Running ComBat...
""")


corrected = neuroCombat(
    dat=combined,
    covars=pd.DataFrame(
        {
            "batch": batch
        }
    ),
    batch_col="batch"
)["data"]



corrected = pd.DataFrame(
    corrected,
    index=combined.index,
    columns=combined.columns
)



# ==========================
# Save
# ==========================

os.makedirs(
    "data",
    exist_ok=True
)


corrected.to_csv(
    OUTPUT
)



print("""
==========================================
FINISHED

Saved:
data/combat_corrected_8gene.csv

==========================================
""")