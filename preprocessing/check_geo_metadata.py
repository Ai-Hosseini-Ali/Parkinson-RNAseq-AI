import pandas as pd
import os

datasets = {
    "GSE7621": "data/GEO/GSE7621/processed/GSE7621_series_matrix.txt",
    "GSE20292": "data/GEO/GSE20292/processed/GSE20292_series_matrix.txt",
    "GSE136666": "data/GEO/GSE136666/processed/GSE136666_series_matrix.txt"
}

for name, path in datasets.items():

    print("\n==========================")
    print(name)

    # خواندن metadata از خطوط !Sample
    metadata = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                metadata.append(line.strip())

    print("Sample characteristics:")
    
    for m in metadata[:5]:
        print(m[:300])

    # خواندن expression matrix
    df = pd.read_csv(
        path,
        sep="\t",
        comment="!",
        index_col=0
    )

    print("\nExpression shape:")
    print(df.shape)

    print("First genes:")
    print(df.index[:5].tolist())