import pandas as pd

bio = pd.read_csv("data/biomarker_genes.csv")
ann = pd.read_csv("data/annotated_candidate_genes.csv")

ann = ann[["Gene", "symbol"]]

bio = bio.merge(ann, on="Gene", how="left")

bio["Gene"] = bio["symbol"].fillna(bio["Gene"])

bio = bio.drop(columns=["symbol"])

bio.to_csv("data/biomarker_genes_symbol.csv", index=False)

print("Saved: data/biomarker_genes_symbol.csv")