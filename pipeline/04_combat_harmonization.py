import pandas as pd
import numpy as np
import json

from combat.pycombat import pycombat


print("=" * 60)
print("COMBAT HARMONIZATION - GSE99039 + GSE165082")
print("=" * 60)


# ==========================
# Load locked genes
# ==========================

with open(
    "pipeline/gene_signature_locked.json",
    encoding="utf-8"
) as f:
    genes = json.load(f)["genes"]


print("\nLocked genes:")
print(genes)


# ==========================
# Load GSE99039
# ==========================

gse99039 = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)


if "Sample" in gse99039.columns:
    gse99039 = gse99039.set_index("Sample")


# remove label column
if "Disease" in gse99039.columns:
    gse99039 = gse99039.drop(
        columns=["Disease"]
    )


gse99039 = gse99039[genes]


print("\nGSE99039:")
print(gse99039.shape)



# ==========================
# Load GSE165082
# ==========================

gse165082 = pd.read_csv(
    "data/GSE165082_normalized_log_cpm.csv",
    index_col=0
)


gse165082 = gse165082[genes]


print("\nGSE165082:")
print(gse165082.shape)



# ==========================
# Combine
# ==========================

combined = pd.concat(
    [
        gse99039,
        gse165082
    ],
    axis=0
)


print("\nCombined:")
print(combined.shape)



# ComBat expects:
# genes x samples

data = combined.T


# ==========================
# Batch labels
# ==========================

batch = (
    ["GSE99039"] * len(gse99039)
    +
    ["GSE165082"] * len(gse165082)
)


print("\nBatch:")
print(
    pd.Series(batch).value_counts()
)



# ==========================
# Run ComBat
# ==========================

print("\nRunning ComBat...")


combat_result = pycombat(
    data,
    batch
)


combat_result = combat_result.T



# ==========================
# Save
# ==========================

combat_result.to_csv(
    "data/combat_harmonized_14genes.csv"
)


print("\nSaved:")
print(
    "data/combat_harmonized_14genes.csv"
)


print("\nFinal shape:")
print(
    combat_result.shape
)