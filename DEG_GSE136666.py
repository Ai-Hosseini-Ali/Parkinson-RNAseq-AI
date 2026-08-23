import pandas as pd
import os

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# Paths

count_file = "data/GEO/GSE136666/processed/GSE136666_SN_counts_matrix.csv"
metadata_file = "data/GEO/GSE136666/processed/GSE136666_metadata.csv"


output_dir = "data/GEO/GSE136666/processed"


# Load counts

counts = pd.read_csv(
    count_file,
    index_col=0
)

print("Counts shape:")
print(counts.shape)


# Load metadata

metadata = pd.read_csv(
    metadata_file
)

print("\nMetadata:")
print(metadata)


# Make metadata index same as count columns

metadata.index = metadata["Sample"]

metadata = metadata.loc[
    counts.columns
]


# Rename condition

metadata["Condition"] = metadata["Condition"].astype(str)


# Convert counts to integer

counts = counts.round().astype(int)


# Transpose
# pydeseq2 needs:
# rows = samples
# columns = genes

counts = counts.T


print("\nFinal counts:")
print(counts.shape)


# DESeq2 dataset

dds = DeseqDataSet(
    counts=counts,
    metadata=metadata,
    design_factors="Condition"
)


print("\nRunning DESeq2...")
dds.deseq2()


# Statistics

stat_res = DeseqStats(
    dds,
    contrast=[
        "Condition",
        "Parkinson",
        "Control"
    ]
)


stat_res.summary()


# Results

results = stat_res.results_df


results = results.sort_values(
    "padj"
)


print("\nTop genes:")
print(results.head(20))


# Save

results.to_csv(
    os.path.join(
        output_dir,
        "GSE136666_DEG_results.csv"
    )
)


print("\nSaved DEG results!")