import pandas as pd
import matplotlib.pyplot as plt

# خواندن فایل Biomarker با Gene Symbol
df = pd.read_csv("data/biomarker_genes_symbol.csv")

# مرتب‌سازی
df = df.sort_values(
    by="Abs_Coefficient",
    ascending=False
)

# انتخاب 20 ژن برتر
top20 = df.head(20)

# رسم نمودار
plt.figure(figsize=(10,8))

bars = plt.barh(
    top20["Gene"],
    top20["Abs_Coefficient"]
)

plt.gca().invert_yaxis()

plt.xlabel("Absolute Logistic Regression Coefficient", fontsize=12)
plt.ylabel("Gene Symbol", fontsize=12)
plt.title("Top 20 Candidate Biomarker Genes", fontsize=14)

# نمایش مقدار ضریب کنار هر میله
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.002,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "data/figure3_top_biomarkers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Figure saved:")
print("data/figure3_top_biomarkers.png")