import pandas as pd


# Files

deg_file = "data/GEO/GSE136666/processed/GSE136666_significant_DEGs.csv"

annotation_file = "data/GEO/GSE136666/processed/GSE136666_SN_gene_symbol_matrix.csv"


# Load DEG

deg = pd.read_csv(
    deg_file,
    index_col=0
)


print("DEG before annotation:")
print(deg.shape)


# Load annotated expression matrix

annot = pd.read_csv(
    annotation_file,
    index_col=0
)


print("\nAnnotation matrix:")
print(annot.shape)


# Check columns

print("\nFirst rows:")
print(annot.head())


# Create Ensembl -> Symbol mapping

mapping = {}

for gene in annot.index:

    # index already contains gene symbols after previous annotation
    mapping[gene] = gene


# Add Gene Symbol column

deg["Gene_ID"] = deg.index


# Try annotation using Ensembl IDs

if "Gene Symbol" not in annot.columns:

    deg["Gene_Symbol"] = deg.index

else:

    gene_map = annot["Gene Symbol"].to_dict()

    deg["Gene_Symbol"] = deg.index.map(gene_map)



# Save

output = "data/GEO/GSE136666/processed/GSE136666_DEG_annotated.csv"


deg.to_csv(output)


print("\nAfter annotation:")
print(deg.head(20))


print("\nSaved:")
print(output)