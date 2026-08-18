import pandas as pd

# ==========================
# Load differential expression with FDR
# ==========================

df = pd.read_csv(
    "differential_expression_GSE99039_FDR.csv"
)

print("Original:")
print(df.shape)

# ==========================
# Keep only significant genes
# ==========================

df = df[df["FDR"] < 0.05]

print("After FDR filtering:")
print(df.shape)

# ==========================
# Sort by FDR
# ==========================

df = df.sort_values("FDR")

# ==========================
# Save Top100
# ==========================

top100 = df.head(100)

top100.to_csv(
    "top100_genes_GSE99039.csv",
    index=False
)

# ==========================
# Save Top300
# ==========================

top300 = df.head(300)

top300.to_csv(
    "top300_genes_GSE99039.csv",
    index=False
)

# ==========================
# Save Top500
# ==========================

top500 = df.head(500)

top500.to_csv(
    "top500_genes_GSE99039.csv",
    index=False
)

print()

print("Top100:", len(top100))
print("Top300:", len(top300))
print("Top500:", len(top500))

print()

print("Top 10 genes:")

print(top100[["Gene","LogFC","FDR"]].head(10))

print("\nSaved successfully.")