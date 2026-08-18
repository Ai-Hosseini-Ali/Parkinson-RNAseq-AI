import pandas as pd
import gseapy as gp
import os


# ==========================
# Load biomarker genes
# ==========================

df = pd.read_csv(
    "data/pathway_input_biomarkers.csv"
)

print("Input:")
print(df.shape)


# ==========================
# Prepare gene list
# ==========================

genes = (
    df["Gene"]
    .dropna()
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)

print("\nNumber of genes:")
print(len(genes))


# ==========================
# GO Biological Process
# ==========================

print("\nRunning GO enrichment...")
go = gp.enrichr(
    gene_list=genes,
    gene_sets=[
        "GO_Biological_Process_2023"
    ],
    organism="human",
    outdir=None
)

go_result = go.results


print("\nTop GO terms:")
print(
    go_result[
        [
            "Term",
            "Adjusted P-value",
            "Combined Score"
        ]
    ].head(20)
)


# ==========================
# KEGG
# ==========================

print("\nRunning KEGG enrichment...")

kegg = gp.enrichr(
    gene_list=genes,
    gene_sets=[
        "KEGG_2021_Human"
    ],
    organism="human",
    outdir=None
)
kegg_result = kegg.results


print("\nTop KEGG pathways:")

print(
    kegg_result[
        [
            "Term",
            "Adjusted P-value",
            "Combined Score"
        ]
    ].head(20)
)


# ==========================
# Save results
# ==========================

os.makedirs(
    "results",
    exist_ok=True
)


go_result.to_csv(
    "results/GO_enrichment_results.csv",
    index=False
)


kegg_result.to_csv(
    "results/KEGG_enrichment_results.csv",
    index=False
)


print("\nSaved successfully")