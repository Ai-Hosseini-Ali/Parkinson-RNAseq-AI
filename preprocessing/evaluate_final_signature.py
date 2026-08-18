import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
# 1. Load expression dataset
# =====================================

expression = pd.read_csv(
    "data/GSE99039_final_dataset.csv"
)

print("Expression:")
print(expression.shape)


print("\nFirst columns:")
print(expression.columns[:5])


# Fix sample column name

if "Unnamed: 0" in expression.columns:
    expression = expression.rename(
        columns={
            "Unnamed: 0": "Sample"
        }
    )


print("\nSample column:")
print(expression["Sample"].head())



# =====================================
# 2. Load labels
# =====================================

labels = pd.read_csv(
    "data/GSE99039_labels.csv"
)


print("\nLabels:")
print(labels.head())



# =====================================
# 3. Merge expression + labels
# =====================================

data = expression.merge(
    labels[["Sample", "Label"]],
    on="Sample",
    how="inner"
)


print("\nMerged dataset:")
print(data.shape)



print("\nClass distribution:")
print(data["Label"].value_counts())



# =====================================
# 4. Load final biomarkers
# =====================================

biomarkers = pd.read_csv(
    "data/final_ranked_biomarkers.csv"
)


top_genes = biomarkers.head(20)["Gene"].tolist()


print("\nSelected genes:")
print(top_genes)



# Check genes exist

genes = [
    g for g in top_genes
    if g in data.columns
]


print("\nAvailable genes:")
print(genes)



if len(genes) == 0:
    raise Exception(
        "No biomarker genes found in expression dataset"
    )



# =====================================
# 5. Prepare ML matrix
# =====================================

X = data[genes]

y = data["Label"]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# =====================================
# 6. PCA visualization
# =====================================

pca = PCA(
    n_components=2
)


pc = pca.fit_transform(
    X_scaled
)



plt.figure(
    figsize=(7,5)
)


plt.scatter(
    pc[:,0],
    pc[:,1],
    c=y,
    alpha=0.7
)


plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "PCA - Parkinson Biomarker Signature"
)


plt.savefig(
    "data/PCA_signature.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================
# 7. Random Forest classification
# =====================================


model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)


model.fit(
    X_scaled,
    y
)



probability = model.predict_proba(
    X_scaled
)[:,1]



auc = roc_auc_score(
    y,
    probability
)


print("\n====================")
print("AUC:")
print(auc)
print("====================")



# =====================================
# 8. ROC Curve
# =====================================


fpr, tpr, _ = roc_curve(
    y,
    probability
)



plt.figure(
    figsize=(6,5)
)


plt.plot(
    fpr,
    tpr
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)


plt.title(
    f"ROC Curve AUC={auc:.3f}"
)


plt.savefig(
    "data/ROC_signature.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================
# 9. Heatmap
# =====================================


plt.figure(
    figsize=(10,8)
)


sns.clustermap(
    data[genes].T,
    cmap="vlag",
    figsize=(10,8)
)


plt.savefig(
    "data/Heatmap_signature.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================
# 10. Save evaluated genes
# =====================================


output = biomarkers[
    biomarkers["Gene"].isin(genes)
]


output.to_csv(
    "data/evaluated_signature_genes.csv",
    index=False
)



print("\nFiles saved:")
print("PCA_signature.png")
print("ROC_signature.png")
print("Heatmap_signature.png")
print("evaluated_signature_genes.csv")

print("\nFinished successfully")