import pandas as pd


files = [
    "data/GSE99039_candidate_biomarkers.csv",
    "data/biomarker_genes.csv",
    "data/candidate_genes.csv"
]


for file in files:

    print("\n===================")
    print(file)

    df = pd.read_csv(file)

    print(df.head(20))
