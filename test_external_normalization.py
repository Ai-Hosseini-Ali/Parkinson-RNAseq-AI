import pandas as pd
import numpy as np


# Load external raw counts

df = pd.read_csv(
    "data/GSE165082_gene_symbol.csv"
)


df = df.set_index(
    "GeneSymbol"
)


# Total counts per sample

library_size = df.sum(axis=0)


print("Library size:")
print(library_size.describe())


# CPM normalization

cpm = df.div(
    library_size,
    axis=1
) * 1e6


# log transform

log_cpm = np.log2(
    cpm + 1
)


for gene in [
    "PTGDS",
    "PPP4C",
    "TYROBP",
    "MMP9"
]:

    print("\n================")
    print(gene)

    print(
        log_cpm.loc[gene].describe()
    )