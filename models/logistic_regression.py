import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# خواندن داده
df = pd.read_csv(
    "data/GSE165082_PD-CC.counts.txt",
    sep="\t"
)

# آماده‌سازی داده
genes = df["Geneid"]

counts = df.drop("Geneid", axis=1)

filtered = counts[counts.sum(axis=1) >= 10]

normalized = np.log2(filtered + 1)

X = normalized.T

# ساخت برچسب‌ها
y = [1 if "PD" in sample else 0 for sample in X.index]
y = pd.Series(y, index=X.index)

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ساخت مدل
model = LogisticRegression(max_iter=5000)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی
y_pred = model.predict(X_test)

# محاسبه Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)