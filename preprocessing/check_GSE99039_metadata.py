import pandas as pd
import os

print("======================")
print("GSE99039 METADATA CHECK")
print("======================")


# files
files = [
    "data/GSE99039_labels.csv",
    "data/GSE99039_final_dataset.csv"
]


for f in files:

    print("\nFILE:")
    print(f)

    if os.path.exists(f):
        df = pd.read_csv(f)

        print("Shape:")
        print(df.shape)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst rows:")
        print(df.head())

    else:
        print("NOT FOUND")



# labels separately

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)

print("\n======================")
print("LABEL INFORMATION")
print("======================")

print(labels.head())

print("\nLabel columns:")
print(labels.columns)


for col in labels.columns:
    print("\n", col)
    print(labels[col].value_counts())