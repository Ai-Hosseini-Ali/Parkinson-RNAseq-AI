# check GPL570 annotation header

with open("data/GPL570.annot", "r", encoding="utf-8", errors="ignore") as f:

    for i in range(40):
        line = f.readline()
        print(i+1, line[:200])
        