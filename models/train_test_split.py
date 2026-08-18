import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


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

# تبدیل نمونه‌ها به سطر
X = normalized.T

# ساخت Label
y = []

for sample in X.index:
    if "PD" in sample:
        y.append(1)
    else:
        y.append(0)

y = pd.Series(y, index=X.index)


# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("X_train shape:")
print(X_train.shape)

print("\nX_test shape:")
print(X_test.shape)

print("\ny_train:")
print(y_train.value_counts())

print("\ny_test:")
print(y_test.value_counts())