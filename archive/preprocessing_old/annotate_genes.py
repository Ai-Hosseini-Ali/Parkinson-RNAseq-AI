import pandas as pd
import mygene


# خواندن candidate genes

df = pd.read_csv(
    "data/candidate_genes.csv"
)


mg = mygene.MyGeneInfo()


genes = df["Gene"].tolist()


# Annotation

result = mg.querymany(
    genes,
    scopes="ensembl.gene",
    fields="symbol,name",
    species="human"
)


annotation = pd.DataFrame(result)


# ترکیب با نتایج مدل

merged = df.merge(
    annotation,
    left_on="Gene",
    right_on="query",
    how="left"
)


merged.to_csv(
    "data/annotated_candidate_genes.csv",
    index=False
)


print(
    merged[
        ["Gene","symbol","name"]
    ].head(20)
)


print("\nSaved:")
print("data/annotated_candidate_genes.csv")