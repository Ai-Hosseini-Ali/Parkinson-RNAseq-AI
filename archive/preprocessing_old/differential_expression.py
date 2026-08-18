import pandas as pd
import numpy as np

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)


genes = df["Geneid"]

counts = df.drop("Geneid", axis=1)


PD_samples = [
    c for c in counts.columns
    if "PD" in c
]


CC_samples = [
    c for c in counts.columns
    if "CC" in c
]


# log normalization
data = np.log2(counts + 1)


results = []


for i, gene in enumerate(genes):

    pd_values = data.loc[i, PD_samples]

    cc_values = data.loc[i, CC_samples]


    log2fc = (
        pd_values.mean()
        -
        cc_values.mean()
    )


    stat, p = ttest_ind(
        pd_values,
        cc_values,
        equal_var=False
    )


    results.append([
        gene,
        log2fc,
        p
    ])


de = pd.DataFrame(
    results,
    columns=[
        "Gene",
        "log2FC",
        "p_value"
    ]
)
# حذف ژن‌هایی که p-value ندارند
de = de.dropna(
    subset=["p_value"]
)

# FDR
de["FDR"] = multipletests(
    de["p_value"],
    method="fdr_bh"
)[1]


# مرتب سازی
de = de.sort_values(
    "FDR"
)


print(de.head(20))


de.to_csv(
    "data/differential_expression.csv",
    index=False
)


print("\nSaved:")
print("data/differential_expression.csv")