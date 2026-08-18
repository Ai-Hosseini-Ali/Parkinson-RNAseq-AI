import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# ==========================
# Load dataset
# ==========================

df = pd.read_csv("data/GSE99039_top100_dataset.csv")

X = df.drop(columns=["Disease"])
y = df["Disease"]

# ==========================
# Train model
# ==========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

model.fit(X, y)

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Gene": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(20))

importance.to_csv(
    "data/GSE99039_feature_importance.csv",
    index=False
)

# ==========================
# Plot
# ==========================

top20 = importance.head(20)

plt.figure(figsize=(8,8))

plt.barh(
    top20["Gene"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("Top 20 Biomarker Genes")

plt.tight_layout()

plt.savefig(
    "data/GSE99039_feature_importance.png",
    dpi=300
)

plt.show()

print("Saved successfully.")