import pandas as pd
import mygene

# load expression
expr = pd.read_csv(
    "data/GEO/GSE136666/processed/GSE136666_SN_counts_matrix.csv",
    index_col=0
)

print("Before annotation:")
print(expr.shape)


# connect to MyGene
mg = mygene.MyGeneInfo()

genes = expr.index.tolist()


# query ENSG
results = mg.querymany(
    genes,
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    returnall=False
)


mapping = {}

for r in results:
    if "symbol" in r:
        mapping[r["query"]] = r["symbol"]


expr["Gene_Symbol"] = expr.index.map(mapping)


# remove genes without symbol
expr = expr.dropna(subset=["Gene_Symbol"])


# remove duplicate symbols
expr = expr.groupby("Gene_Symbol").mean()


print("After annotation:")
print(expr.shape)


expr.to_csv(
    "data/GEO/GSE136666/processed/GSE136666_SN_gene_symbol_matrix.csv"
)

print("Saved!")