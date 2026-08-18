import pandas as pd
import gseapy as gp


# خواندن ژن‌های annotation شده

df = pd.read_csv(
    "data/annotated_candidate_genes.csv"
)


# حذف ژن‌هایی که Symbol ندارند
genes = (
    df.sort_values(
        "Abs_Coefficient",
        ascending=False
    )
    ["symbol"]
    .dropna()
    .unique()
    .tolist()[:50]
)

print("Number of genes:")
print(len(genes))


print("\nGenes:")
print(genes[:20])


# =========================
# GO Enrichment
# =========================

enr = gp.enrichr(
    gene_list=genes,
    gene_sets=[
        "GO_Biological_Process_2023",
        "KEGG_2021_Human",
        "Reactome_2022"
    ],
    organism="human",
    outdir=None,
    cutoff=0.5
)
# نمایش نتایج

for library in enr.results["Gene_set"].unique():

    print("\n===================")
    print(library)
    print("===================")

    result = enr.results[
        enr.results["Gene_set"] == library
    ]

    print(
        result[
            [
                "Term",
                "Adjusted P-value"
            ]
        ]
        .head(10)
    )


print("\nSaved:")
print("data/pathway_results")
for library in enr.results["Gene_set"].unique():

    print("\n===================")
    print(library)
    print("===================")

    result = enr.results[
        enr.results["Gene_set"] == library
    ]

    if len(result) == 0:
        print("No enriched terms")
    else:
        print(
            result[
                [
                    "Term",
                    "Adjusted P-value"
                ]
            ].head(10)
        )