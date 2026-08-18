import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load data
# ==========================

df = pd.read_csv("differential_expression_GSE99039_FDR.csv")

# ==========================
# -log10(FDR)
# ==========================

df["minusLog10FDR"] = -np.log10(df["FDR"])

# ==========================
# Define significance
# ==========================

df["Category"] = "Not Significant"

df.loc[
    (df["FDR"] < 0.05) & (df["LogFC"] > 0.2),
    "Category"
] = "Upregulated"

df.loc[
    (df["FDR"] < 0.05) & (df["LogFC"] < -0.2),
    "Category"
] = "Downregulated"

print(df["Category"].value_counts())

# ==========================
# Plot
# ==========================

plt.figure(figsize=(10,8))

colors = {
    "Not Significant":"lightgray",
    "Upregulated":"red",
    "Downregulated":"blue"
}

for cat in colors:

    subset = df[df["Category"] == cat]

    plt.scatter(
        subset["LogFC"],
        subset["minusLog10FDR"],
        s=10,
        c=colors[cat],
        label=cat,
        alpha=0.7
    )

plt.axhline(
    -np.log10(0.05),
    color="black",
    linestyle="--"
)

plt.axvline(
    0.2,
    color="black",
    linestyle="--"
)

plt.axvline(
    -0.2,
    color="black",
    linestyle="--"
)

plt.xlabel("Log Fold Change")
plt.ylabel("-log10(FDR)")
plt.title("Volcano Plot - GSE99039")

plt.legend()

plt.tight_layout()

plt.savefig(
    "figure_GSE99039_volcano.png",
    dpi=300
)

plt.show()

print("\nFigure saved successfully.")