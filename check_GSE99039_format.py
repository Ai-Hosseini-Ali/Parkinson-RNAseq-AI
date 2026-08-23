import pandas as pd

file="data/GSE99039_gene_expression.csv"

df=pd.read_csv(file,index_col=0)

print(df.shape)

print(df.head())

print(df.index[:20])