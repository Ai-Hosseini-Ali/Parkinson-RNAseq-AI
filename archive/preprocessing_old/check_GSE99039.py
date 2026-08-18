import pandas as pd
import os


# ==========================
# File path
# ==========================

file_path = "data/GSE99039_series_matrix.txt"


# ==========================
# Check file exists
# ==========================

if not os.path.exists(file_path):
    print("❌ File not found:")
    print(file_path)
    exit()

print("✅ File found")


# ==========================
# Read GEO Series Matrix
# ==========================

df = pd.read_csv(
    file_path,
    sep="\t",
    comment="!",
    low_memory=False
)


# ==========================
# Basic information
# ==========================

print("\n==========================")
print("Dataset Information")
print("==========================")

print("Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns[:10])


print("\nFirst 5 rows:")
print(df.head())


# ==========================
# Check index / probe IDs
# ==========================

print("\nData types:")
print(df.dtypes.head())


print("\nNumber of probes:")
print(len(df))


print("\nNumber of samples:")
print(df.shape[1] - 1)
print("\nLast columns:")
print(df.columns[-5:])