import os

path = os.path.join(
    os.path.dirname(__file__),
    "final_genes.txt"
)

print("Loading:", path)

with open(path, "r") as f:
    genes = [line.strip() for line in f.readlines() if line.strip()]

print("\nNumber of genes:", len(genes))

print("\nGenes:")
for i, gene in enumerate(genes, 1):
    print(i, gene)