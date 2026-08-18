import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


print("\n===================================")
print(" EXTERNAL PROBABILITY DISTRIBUTION ")
print(" GSE165082")
print("===================================\n")


# -------------------------------
# Load probabilities
# -------------------------------

file = "results/GSE165082_probabilities.csv"

df = pd.read_csv(file)

print(df.head())

print("\nClass distribution:")
print(df["True_Label"].value_counts())


# -------------------------------
# Separate groups
# -------------------------------

pd_group = df[df["True_Label"] == 1]["Probability_PD"]

cc_group = df[df["True_Label"] == 0]["Probability_PD"]


print("\nPD probability statistics")
print(pd_group.describe())


print("\nControl probability statistics")
print(cc_group.describe())



# -------------------------------
# Mean comparison
# -------------------------------

print("\nMean probability")

print(
    "PD:",
    pd_group.mean()
)

print(
    "Control:",
    cc_group.mean()
)



# -------------------------------
# Histogram
# -------------------------------

plt.figure(figsize=(8,5))


plt.hist(
    cc_group,
    bins=10,
    alpha=0.7,
    label="Control"
)


plt.hist(
    pd_group,
    bins=10,
    alpha=0.7,
    label="Parkinson"
)


plt.xlabel(
    "Predicted Probability of Parkinson"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "GSE165082 External Probability Distribution"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "results/GSE165082_probability_distribution.png",
    dpi=300
)


plt.close()



# -------------------------------
# Threshold analysis
# -------------------------------

print("\nThreshold analysis")

for threshold in [0.01,0.02,0.05,0.1,0.2,0.3,0.5]:

    pred = (
        df["Probability_PD"] >= threshold
    ).astype(int)


    TP = sum(
        (pred==1)&
        (df["True_Label"]==1)
    )

    FP = sum(
        (pred==1)&
        (df["True_Label"]==0)
    )

    FN = sum(
        (pred==0)&
        (df["True_Label"]==1)
    )

    TN = sum(
        (pred==0)&
        (df["True_Label"]==0)
    )


    sensitivity = TP/(TP+FN)

    specificity = TN/(TN+FP)


    print(
        f"Threshold {threshold}: "
        f"Sensitivity={sensitivity:.3f}, "
        f"Specificity={specificity:.3f}"
    )



# -------------------------------
# ROC
# -------------------------------

fpr, tpr, thresholds = roc_curve(
    df["True_Label"],
    df["Probability_PD"]
)


roc_auc = auc(
    fpr,
    tpr
)


plt.figure(figsize=(6,6))


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
    f"GSE165082 ROC AUC={roc_auc:.3f}"
)


plt.savefig(
    "results/GSE165082_external_ROC.png",
    dpi=300
)


plt.close()



# -------------------------------
# Save summary
# -------------------------------

summary = pd.DataFrame({

"Group":[
"Parkinson",
"Control"
],

"Mean_probability":[
pd_group.mean(),
cc_group.mean()
],

"Median_probability":[
pd_group.median(),
cc_group.median()
],

"Std":[
pd_group.std(),
cc_group.std()
]

})


summary.to_csv(
"results/GSE165082_probability_summary.csv",
index=False
)



print("\nSaved:")
print("results/GSE165082_probability_distribution.png")
print("results/GSE165082_external_ROC.png")
print("results/GSE165082_probability_summary.csv")

print("\nDONE")