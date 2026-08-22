import pandas as pd
import gzip
import os


print("""
================================
ANNOTATE GSE49036
GPL570
================================
""")


# ==========================
# Paths
# ==========================

EXPRESSION_FILE = "data/GSE49036_series_matrix.txt"
ANNOTATION_FILE = "data/GPL570.annot.gz"

OUTPUT_FILE = "data/GSE49036_gene_expression.csv"



# ==========================
# Load expression data
# ==========================

print("\nLoading expression...")


expr = pd.read_csv(
    EXPRESSION_FILE,
    sep="\t",
    comment="!"
)


print("Raw expression:")
print(expr.shape)


# Rename probe column

expr = expr.rename(
    columns={
        expr.columns[0]: "ID"
    }
)


print("\nExpression columns:")
print(expr.columns[:5])



# ==========================
# Find annotation header
# ==========================

print("\nFinding annotation header...")


header_line = None


with gzip.open(
    ANNOTATION_FILE,
    "rt",
    encoding="utf-8",
    errors="ignore"
) as f:

    for i,line in enumerate(f):

        if line.startswith("ID\t"):

            header_line = i
            break



print("Annotation header:")
print(header_line)



if header_line is None:
    raise Exception(
        "Annotation header not found"
    )



# ==========================
# Load annotation
# ==========================

print("\nLoading annotation...")


annot = pd.read_csv(
    ANNOTATION_FILE,
    sep="\t",
    skiprows=header_line,
    compression="gzip",
    low_memory=False
)



print("Annotation:")
print(annot.shape)


print("\nAnnotation columns:")
print(
    annot.columns[:10].tolist()
)



# ==========================
# Keep required columns
# ==========================

annot = annot[
    [
        "ID",
        "Gene symbol"
    ]
]


# remove empty symbols

annot = annot[
    annot["Gene symbol"].notna()
]


annot = annot[
    annot["Gene symbol"] != ""
]



# ==========================
# Merge probe with genes
# ==========================

print("\nMerging...")


merged = expr.merge(
    annot,
    on="ID",
    how="inner"
)



print("Merged:")
print(merged.shape)



# ==========================
# Probe -> Gene
# ==========================


print("\nConverting probes to genes...")


gene_matrix = (
    merged
    .drop(columns=["ID"])
    .groupby("Gene symbol")
    .mean(numeric_only=True)
)



print("\nGene expression matrix:")
print(gene_matrix.shape)



# ==========================
# Check 8 genes
# ==========================


final_genes = [

    "PTGDS",
    "FAM102A",
    "CHST15",
    "KIR2DL1",
    "DGKK",
    "KIR3DL1",
    "KIR2DL3",
    "LAT2"

]


print("\nChecking final 8 genes:")
print("--------------------------------")


found = []

for gene in final_genes:

    if gene in gene_matrix.index:
        print(gene, "FOUND")
        found.append(gene)

    else:
        print(gene, "MISSING")



print("\nFound:")
print(
    len(found),
    "/ 8"
)



# ==========================
# Save
# ==========================


os.makedirs(
    "data",
    exist_ok=True
)


gene_matrix.to_csv(
    OUTPUT_FILE
)


print("""
================================
FINISHED

Saved:
data/GSE49036_gene_expression.csv

================================
""")