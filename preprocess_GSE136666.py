import pandas as pd
import numpy as np
import os

input_file = "data/GEO/GSE136666/processed/GSE136666_SN_gene_symbol_matrix.csv"

output_dir = "data/GEO/GSE136666/processed"


# Load data
df = pd.read_csv(input_file, index_col=0)

print("Original shape:")
print(df.shape)


# remove genes with too many zeros
threshold = int(df.shape[1] * 0.5)

df_filtered = df[
    (df > 0).sum(axis=1) >= threshold
]


print("\nAfter filtering:")
print(df_filtered.shape)


# log2 transformation
df_log = np.log2(df_filtered + 1)


print("\nAfter log2:")
print(df_log.shape)


# save

df_log.to_csv(
    os.path.join(
        output_dir,
        "GSE136666_log_expression.csv"
    )
)


print("\nSaved!")