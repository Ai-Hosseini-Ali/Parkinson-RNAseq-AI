import pandas as pd
import numpy as np


# خواندن داده
df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)


# جدا کردن ژن‌ها
genes = df["Geneid"]

counts = df.drop("Geneid", axis=1)


# Filtering
filtered = counts[counts.sum(axis=1) >= 10]


# Normalization
normalized = np.log2(filtered + 1)


# گروه‌ها
PD_samples = [col for col in normalized.columns if "PD" in col]
CC_samples = [col for col in normalized.columns if "CC" in col]


# میانگین هر گروه
PD_mean = normalized[PD_samples].mean(axis=1)

CC_mean = normalized[CC_samples].mean(axis=1)


# اختلاف بیان
difference = PD_mean - CC_mean


# ساخت جدول
feature_table = pd.DataFrame({
    "Gene": genes[filtered.index],
    "PD_mean": PD_mean,
    "CC_mean": CC_mean,
    "Difference": difference
})


# مرتب کردن
feature_table = feature_table.sort_values(
    by="Difference",
    key=abs,
    ascending=False
)


# انتخاب 500 ژن برتر
top_genes = feature_table.head(500)


print("Top genes:")
print(top_genes.head(20))


print("\nNumber of selected genes:")
print(top_genes.shape)# انتخاب 500 ژن برتر
top_genes = feature_table.head(500)


print("Top genes:")
print(top_genes.head(20))


print("\nNumber of selected genes:")
print(top_genes.shape)


# ذخیره خروجی برای مدل
top_genes.to_csv(
    "data/top500_genes.csv",
    index=False
)


print("\nSaved:")
print("data/top500_genes.csv")