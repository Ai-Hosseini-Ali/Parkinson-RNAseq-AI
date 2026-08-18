import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# خواندن نتایج Differential Expression
df = pd.read_csv("data/differential_expression.csv")

# حذف مقادیر NaN
df = df.dropna(subset=["log2FC", "p_value"])

# محاسبه -log10(p-value)
df["minus_log10_p"] = -np.log10(df["p_value"])

# آستانه‌ها
fc_threshold = 1
p_threshold = 0.05

# تعیین رنگ نقاط
colors = []

for _, row in df.iterrows():

    if row["p_value"] < p_threshold and row["log2FC"] > fc_threshold:
        colors.append("red")

    elif row["p_value"] < p_threshold and row["log2FC"] < -fc_threshold:
        colors.append("blue")

    else:
        colors.append("gray")

# رسم نمودار
plt.figure(figsize=(9,7))

plt.scatter(
    df["log2FC"],
    df["minus_log10_p"],
    c=colors,
    alpha=0.7,
    s=20
)

# خطوط آستانه
plt.axvline(fc_threshold, linestyle="--")
plt.axvline(-fc_threshold, linestyle="--")
plt.axhline(-np.log10(p_threshold), linestyle="--")

plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot of Differentially Expressed Genes")

# نمایش نام 10 ژن با کوچک‌ترین p-value
top = df.nsmallest(10, "p_value")

for _, row in top.iterrows():
    plt.text(
        row["log2FC"],
        row["minus_log10_p"],
        row["Gene"],
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "data/figure4_volcano.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure saved:")
print("data/figure4_volcano.png")