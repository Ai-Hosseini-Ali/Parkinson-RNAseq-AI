import pandas as pd


file="data/GSE49036_series_matrix.txt"


with open(file,"r") as f:

    for line in f:

        if "!Series_platform_id" in line:
            print(line)

        if "!Sample_characteristics_ch1" in line:
            print(line)
            break