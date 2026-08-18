import pandas as pd
import matplotlib.pyplot as plt


# Load stability results
df = pd.read_csv(
    "data/gene_stability_results.csv"
)


# Select top 20 stable genes
top = df.head(20)


plt.figure(figsize=(10,6))

plt.bar(
    top["Gene"],
    top["Stability_%"]
)


plt.xticks(
    rotation=90
)

plt.xlabel(
    "Genes"
)

plt.ylabel(
    "Stability (%)"
)

plt.title(
    "Top 20 Stable Genes from Stability Selection"
)


plt.tight_layout()


plt.savefig(
    "figures/gene_stability_results.png",
    dpi=300
)


plt.show()


print("Saved:")
print("figures/gene_stability_results.png")