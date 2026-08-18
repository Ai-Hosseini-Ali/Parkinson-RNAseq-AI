import pandas as pd
import numpy as np


genes=[
"PTGDS",
"PPP4C",
"TYROBP",
"MMP9"
]


# training

train=pd.read_csv(
"data/GSE99039_final_dataset.csv"
)


print("TRAIN")
print(train[genes].mean())


# external

ext=pd.read_csv(
"data/GSE165082_gene_symbol.csv"
)

ext=ext.set_index("GeneSymbol")


lib=ext.sum(axis=0)

cpm=ext.div(lib)*1e6

log_cpm=np.log2(cpm+1)


print("\nEXTERNAL")
print(log_cpm.loc[genes].mean())