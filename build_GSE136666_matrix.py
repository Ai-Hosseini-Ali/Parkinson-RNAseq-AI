import pandas as pd
import gzip
import os

folder="data/GEO/GSE136666/raw"

files=[
"GSM4054763_Counts_SN_C1.txt.gz",
"GSM4054764_Counts_SN_C2.txt.gz",
"GSM4054765_Counts_SN_C3.txt.gz",
"GSM4054766_Counts_SN_C4.txt.gz",
"GSM4054767_Counts_SN_C5.txt.gz",
"GSM4054768_Counts_SN_PD1.txt.gz",
"GSM4054769_Counts_SN_PD2.txt.gz",
"GSM4054770_Counts_SN_PD3.txt.gz",
"GSM4054771_Counts_SN_PD4.txt.gz",
"GSM4054772_Counts_SN_PD5.txt.gz"
]


expression={}

for file in files:

    path=os.path.join(folder,file)

    genes=[]
    counts=[]

    with gzip.open(path,'rt') as f:

        lines=f.readlines()[1:]

        for line in lines:

            parts=line.strip().split()

            if len(parts)>=2:
                genes.append(parts[0])
                counts.append(float(parts[1]))

    expression[file.replace(".txt.gz","")] = counts


df=pd.DataFrame(expression,index=genes)


print("Shape:")
print(df.shape)

print("\nFirst genes:")
print(df.head())


df.to_csv(
"data/GEO/GSE136666/processed/GSE136666_SN_counts_matrix.csv"
)

print("\nSaved!")