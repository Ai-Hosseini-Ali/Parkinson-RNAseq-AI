import pandas as pd

# ==========================
# Load Probe-Gene expression
# ==========================

df = pd.read_csv(
    "data/GSE99039_probe_gene_expression.csv"
)

print("Original shape:")
print(df.shape)

# ==========================
# Keep first gene if multiple
# مثال:
# MIR4640///DDR1 -> MIR4640
# ==========================

df["Gene"] = df["Gene"].astype(str).str.split("///").str[0]

# حذف ژن‌های خالی
df = df[df["Gene"] != "nan"]

# ==========================
# Remove Probe column
# ==========================

df = df.drop(columns=["Probe"])

# ==========================
# Average duplicate genes
# ==========================

gene_expression = df.groupby("Gene").mean()

print("\nGene matrix shape:")
print(gene_expression.shape)

# ==========================
# Save
# ==========================

gene_expression.to_csv(
    "data/GSE99039_gene_expression.csv"
)

print("\nSaved:")
print("data/GSE99039_gene_expression.csv")