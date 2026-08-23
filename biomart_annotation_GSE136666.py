import pandas as pd
import mygene


# Input DEG file
deg_file = "data/GEO/GSE136666/processed/GSE136666_significant_DEGs.csv"


# Output
output_file = "data/GEO/GSE136666/processed/GSE136666_DEG_final_annotation.csv"


# Load DEG results

deg = pd.read_csv(
    deg_file,
    index_col=0
)

print("Input DEG:")
print(deg.shape)


# Get Ensembl IDs

ensembl_ids = list(deg.index)


# MyGene annotation

mg = mygene.MyGeneInfo()


print("Querying Ensembl IDs...")


results = mg.querymany(
    ensembl_ids,
    scopes="ensembl.gene",
    fields="symbol,name",
    species="human",
    returnall=False
)


# Convert result

annotation = []

for r in results:

    annotation.append({
        "Ensembl_ID": r.get("query"),
        "Gene_Symbol": r.get("symbol"),
        "Gene_Name": r.get("name")
    })


annotation_df = pd.DataFrame(annotation)


# Merge

deg["Ensembl_ID"] = deg.index


final = deg.merge(
    annotation_df,
    on="Ensembl_ID",
    how="left"
)


# Save

final.to_csv(
    output_file,
    index=False
)


print("\nFinal annotation:")
print(final.head(20))


print("\nMissing symbols:")
print(
    final["Gene_Symbol"].isna().sum()
)


print("\nSaved:")
print(output_file)