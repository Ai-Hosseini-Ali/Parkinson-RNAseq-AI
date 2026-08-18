import pandas as pd


# ML genes
ml = pd.read_csv(
    "data/biomarker_genes.csv"
)


# Differential expression
de = pd.read_csv(
    "data/differential_expression.csv"
)


# انتخاب ژن‌های با p-value بهتر
de_filtered = de[
    de["p_value"] < 0.05
]


# اشتراک
candidate = ml.merge(
    de_filtered,
    on="Gene"
)


# مرتب سازی
candidate = candidate.sort_values(
    by="Abs_Coefficient",
    ascending=False
)


print("\nCandidate Parkinson Genes:")
print(candidate.head(20))


candidate.to_csv(
    "data/candidate_genes.csv",
    index=False
)


print("\nSaved:")
print("data/candidate_genes.csv")