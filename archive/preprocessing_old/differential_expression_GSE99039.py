import pandas as pd
from scipy.stats import ttest_ind


# =========================
# Load final dataset
# =========================

df = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)

print("Dataset:")
print(df.shape)


# =========================
# Split groups
# =========================

control = df[
    df["Disease"]=="CONTROL"
].copy()

ipd = df[
    df["Disease"]=="IPD"
].copy()
print("CONTROL:", control.shape)
print("IPD:", ipd.shape)


# =========================
# Gene list
# =========================


# فقط ستون‌های عددی (ژن‌ها)

numeric_cols = df.select_dtypes(
    include=["number"]
).columns


genes = list(numeric_cols)

print("Number of genes:")
print(len(genes))

results = []


# =========================
# t-test
# =========================

for gene in genes:

    stat, p = ttest_ind(
        ipd[gene],
        control[gene],
        equal_var=False
    )

    logfc = (
        ipd[gene].mean()
        -
        control[gene].mean()
    )

    results.append(
        [
            gene,
            logfc,
            p
        ]
    )


# =========================
# Save
# =========================

result = pd.DataFrame(
    results,
    columns=[
        "Gene",
        "LogFC",
        "PValue"
    ]
)


result = result.sort_values(
    "PValue"
)


print(result.head())


result.to_csv(
    "data/GSE99039_differential_expression.csv",
    index=False
)


print("\nSaved successfully")