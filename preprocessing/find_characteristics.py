file_path = "data/GSE99039_series_matrix.txt"


with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

    for i, line in enumerate(f):

        if "!Sample_characteristics_ch1" in line:

            print("\nLINE NUMBER:", i)

            print(line[:1000])

            print("-"*80)