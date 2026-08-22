import pandas as pd
import numpy as np

# ============================================================
# STABILITY + SHAP INTEGRATION
# GSE99039
# ============================================================

print("\n==========================================")
print("STABILITY + SHAP INTEGRATION")
print("GSE99039")
print("==========================================\n")


# ============================================================
# Paths
# ============================================================

stability_file = "data/nested_cv_gene_stability.csv"
shap_file = "data/shap_nested_summary.csv"


# ============================================================
# Load
# ============================================================

stability = pd.read_csv(stability_file)
shap = pd.read_csv(shap_file)


print("Stability dataset:")
print(stability.shape)

print("\nSHAP dataset:")
print(shap.shape)


# ============================================================
# Prepare SHAP data
# ============================================================

# Mean SHAP across models for each gene
shap_gene = (
    shap
    .groupby("Gene")
    .agg(
        Mean_SHAP=("Mean_SHAP", "mean"),
        SHAP_Std=("Mean_SHAP", "std"),
        Models=("Model", "nunique"),
        Folds=("Folds", "max")
    )
    .reset_index()
)


# ------------------------------------------------------------
# Separate Logistic and SVM SHAP
# ------------------------------------------------------------

shap_pivot = (
    shap
    .pivot_table(
        index="Gene",
        columns="Model",
        values="Mean_SHAP",
        aggfunc="mean"
    )
    .reset_index()
)


shap_pivot.columns.name = None


# Rename columns if available

if "Logistic" in shap_pivot.columns:
    shap_pivot = shap_pivot.rename(
        columns={
            "Logistic": "SHAP_Logistic"
        }
    )

if "SVM" in shap_pivot.columns:
    shap_pivot = shap_pivot.rename(
        columns={
            "SVM": "SHAP_SVM"
        }
    )


# ============================================================
# Merge
# ============================================================

combined = stability.merge(
    shap_gene,
    on="Gene",
    how="left"
)


combined = combined.merge(
    shap_pivot,
    on="Gene",
    how="left"
)


# ============================================================
# SHAP consistency
# ============================================================

if (
    "SHAP_Logistic" in combined.columns
    and
    "SHAP_SVM" in combined.columns
):

    combined["SHAP_Consistency"] = (
        1 -
        abs(
            combined["SHAP_Logistic"]
            -
            combined["SHAP_SVM"]
        )
        /
        (
            combined[
                [
                    "SHAP_Logistic",
                    "SHAP_SVM"
                ]
            ].max(axis=1)
            + 1e-8
        )
    )

else:

    combined["SHAP_Consistency"] = np.nan


# ============================================================
# Normalize Stability
# ============================================================

combined["Stability_Score"] = (
    combined["Selection_%"] / 100
)


# ============================================================
# Normalize SHAP
# ============================================================

max_shap = combined["Mean_SHAP"].max()

combined["SHAP_Score"] = (
    combined["Mean_SHAP"] / max_shap
)


# ============================================================
# Combined Score
# ============================================================

# Equal contribution:
#
# 50% Stability
# 50% SHAP
#
# This is NOT the final biomarker score.
# It is only a candidate ranking.

combined["Combined_Score"] = (
    0.5 * combined["Stability_Score"]
    +
    0.5 * combined["SHAP_Score"]
)


# ============================================================
# Sort
# ============================================================

combined = combined.sort_values(
    "Combined_Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# Rank
# ============================================================

combined["Rank"] = (
    combined.index + 1
)


# ============================================================
# Reorder columns
# ============================================================

preferred_columns = [
    "Rank",
    "Gene",
    "Selection_%",
    "Selected_Folds",
    "Mean_SHAP",
    "SHAP_Logistic",
    "SHAP_SVM",
    "SHAP_Consistency",
    "Stability_Score",
    "SHAP_Score",
    "Combined_Score",
    "Models",
    "Folds"
]


existing_columns = [
    c for c in preferred_columns
    if c in combined.columns
]


remaining_columns = [
    c for c in combined.columns
    if c not in existing_columns
]


combined = combined[
    existing_columns + remaining_columns
]


# ============================================================
# Print results
# ============================================================

print("\n==========================================")
print("INTEGRATED GENE RANKING")
print("==========================================\n")

print(
    combined[
        [
            "Rank",
            "Gene",
            "Selection_%",
            "Mean_SHAP",
            "SHAP_Consistency",
            "Combined_Score"
        ]
    ].head(30)
)


# ============================================================
# Candidate groups
# ============================================================

high_stability = combined[
    combined["Selection_%"] >= 80
].copy()


high_stability = high_stability.sort_values(
    "Combined_Score",
    ascending=False
)


print("\n==========================================")
print("HIGH-STABILITY GENES (>=80%)")
print("==========================================\n")

print(
    high_stability[
        [
            "Gene",
            "Selection_%",
            "Mean_SHAP",
            "Combined_Score"
        ]
    ]
)


# ============================================================
# Save
# ============================================================

combined.to_csv(
    "data/stability_shap_integrated.csv",
    index=False
)


high_stability.to_csv(
    "data/high_stability_genes.csv",
    index=False
)


print("\n==========================================")
print("FINISHED")
print("==========================================")

print("\nSaved:")
print("data/stability_shap_integrated.csv")
print("data/high_stability_genes.csv")