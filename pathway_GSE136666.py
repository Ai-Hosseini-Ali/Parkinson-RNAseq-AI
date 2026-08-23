import pandas as pd
from gprofiler import GProfiler


# Input

input_file = "data/GEO/GSE136666/processed/GSE136666_DEG_final_annotation.csv"


# Output

output_file = "data/GEO/GSE136666/processed/GSE136666_pathway_enrichment.csv"



# Load DEG

df = pd.read_csv(input_file)


print("Input DEG:")
print(df.shape)


# Remove missing symbols

genes = (
    df["Gene_Symbol"]
    .dropna()
    .unique()
    .tolist()
)


print("\nGenes used:")
print(len(genes))

print(genes)



# gProfiler

gp = GProfiler(
    return_dataframe=True
)


print("\nRunning enrichment...")


results = gp.profile(
    organism="hsapiens",
    query=genes,
    sources=[
        "GO:BP",
        "KEGG",
        "REAC"
    ],
    user_threshold=0.05
)



# Save

results.to_csv(
    output_file,
    index=False
)


print("\nTop pathways:")
print("\nAvailable columns:")
print(results.columns.tolist())


print("\nTop pathways:")

print(
    results[
        [
            "source",
            "name",
            "p_value",
            "intersection_size"
        ]
    ].head(20)
)


print("\nSaved:")
print(output_file)