import pandas as pd


file_path = "data/GSE99039_series_matrix.txt"


# ==========================
# Sample IDs
# ==========================

samples = []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

    for line in f:

        if line.startswith("!Sample_geo_accession"):

            samples = line.strip().split("\t")[1:]
            break



# ==========================
# Disease labels (LINE 40)
# ==========================

disease = []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

    for i, line in enumerate(f):

        if i == 40:

            disease = line.strip().split("\t")[1:]
            break



print("Samples:", len(samples))
print("Disease:", len(disease))


# ==========================
# dataframe
# ==========================

df = pd.DataFrame(
    {
        "Sample": samples,
        "Disease": disease
    }
)


df["Sample"] = df["Sample"].str.replace('"',"")
df["Disease"] = df["Disease"].str.replace('"',"")


df["Disease"] = (
    df["Disease"]
    .str.replace(
        "disease label: ",
        "",
        regex=False
    )
)


print("\nGroups:")
print(df["Disease"].value_counts())



# ==========================
# Keep PD and Control
# ==========================

df = df[
    df["Disease"].isin(
        [
            "CONTROL",
            "IPD"
        ]
    )
]


df["Label"] = df["Disease"].map(
    {
        "CONTROL":0,
        "IPD":1
    }
)


print("\nFinal:")
print(df["Disease"].value_counts())


print("\nShape:")
print(df.shape)


df.to_csv(
    "data/GSE99039_labels.csv",
    index=False
)


print("\nSaved")