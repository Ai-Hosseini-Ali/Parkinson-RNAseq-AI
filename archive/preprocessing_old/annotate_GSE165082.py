import pandas as pd
import mygene

print("Loading counts...")

df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t",
    index_col=0
)

print(df.shape)

genes = df.index.tolist()

print("Mapping genes...")


mg = mygene.MyGeneInfo()

result = mg.querymany(
    genes,
    scopes="ensembl.gene",
    fields="symbol",
    species="human"
)


mapping = {}

for r in result:
    if "symbol" in r:
        mapping[r["query"]] = r["symbol"]


print(
    "Mapped:",
    len(mapping)
)


df["GeneSymbol"] = [
    mapping.get(x, None)
    for x in df.index
]


df = df.dropna(
    subset=["GeneSymbol"]
)


df = df.set_index(
    "GeneSymbol"
)


# حذف ژن های تکراری
df = df.groupby(
    df.index
).sum()


print(df.shape)


df.to_csv(
    "data/GSE165082_gene_symbol.csv"
)


print("Saved")