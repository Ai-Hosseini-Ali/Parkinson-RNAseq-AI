import pandas as pd
import numpy as np

# خواندن داده
df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)

# جدا کردن Gene ID
genes = df["Geneid"]

# فقط داده‌های شمارشی
counts = df.drop("Geneid", axis=1)

# حذف ژن‌های کم‌بیان
filtered = counts[counts.sum(axis=1) >= 10]

# Log normalization
normalized = np.log2(filtered + 1)

# Transpose
X = normalized.T

# نام ستون‌ها = نام ژن‌ها
X.columns = genes[filtered.index]

print("Shape after transpose:")
print(X.shape)

print("\nFirst rows:")
print(X.head())
# ساخت Label ها
y = []

for sample in X.index:
    if "PD" in sample:
        y.append(1)
    else:
        y.append(0)

# تبدیل به Series
y = pd.Series(y, index=X.index, name="Label")

print("\nLabels:")
print(y)

print("\nNumber of samples in each class:")
print(y.value_counts())