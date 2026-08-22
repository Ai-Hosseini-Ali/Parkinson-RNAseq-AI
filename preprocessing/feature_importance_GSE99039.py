import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ==========================
# Load dataset
# ==========================

df = pd.read_csv(
    "data/GSE99039_top100_dataset.csv"
)

print("Dataset:")
print(df.shape)


X = df.drop(columns=["Disease"])
y = df["Disease"]


# ==========================
# Train Random Forest
# ==========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Gene": X.columns,
    "Importance": model.feature_importances_
})


# IMPORTANT:
# Highest importance first

importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


print("\nTop 20 important genes:")
print(
    importance.head(20)
)


# ==========================
# Save
# ==========================

importance.to_csv(
    "data/GSE99039_feature_importance.csv",
    index=False
)


# ==========================
# Plot
# ==========================

top20 = importance.head(20)

plt.figure(figsize=(8, 8))

plt.barh(
    top20["Gene"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Random Forest Importance")
plt.ylabel("Gene")
plt.title(
    "Top 20 Biomarker Genes - GSE99039"
)

plt.tight_layout()

plt.savefig(
    "data/GSE99039_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved successfully:")
print("data/GSE99039_feature_importance.csv")
print("data/GSE99039_feature_importance.png")