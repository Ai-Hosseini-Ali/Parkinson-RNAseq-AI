import pandas as pd


file_path = "data/GSE99039_series_matrix.txt"


with open(file_path, "r") as f:

    for line in f:

        if "disease" in line.lower():
            print(line.strip())

        if "diagnosis" in line.lower():
            print(line.strip())

        if "group" in line.lower():
            print(line.strip())

        if "status" in line.lower():
            print(line.strip())