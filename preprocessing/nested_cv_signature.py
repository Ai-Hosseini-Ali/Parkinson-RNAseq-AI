import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# NESTED / LEAKAGE-CONTROLLED CROSS VALIDATION
# GSE99039
# ============================================================

print("\n==========================================")
print("NESTED / LEAKAGE-CONTROLLED VALIDATION")
print("GSE99039")
print("==========================================\n")


# ============================================================
# Load dataset
# ============================================================

expression_file = "data/GSE99039_top100_dataset.csv"

data = pd.read_csv(expression_file)

print("Dataset:")
print(data.shape)


# ============================================================
# Separate X / y
# ============================================================

if "Disease" not in data.columns:
    raise ValueError("Disease column not found.")


X = data.drop(columns=["Disease"]).copy()

y = data["Disease"].map({
    "CONTROL": 0,
    "IPD": 1
})


if y.isna().any():
    raise ValueError(
        "Unknown class detected in Disease column."
    )


print("\nNumber of genes:")
print(X.shape[1])


print("\nClasses:")
print(y.value_counts())


# ============================================================
# Outer Cross Validation
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# Models
# ============================================================

models = {

    "SVM": Pipeline([
        (
            "scale",
            StandardScaler()
        ),

        (
            "model",
            SVC(
                kernel="linear",
                probability=True,
                random_state=42
            )
        )
    ]),

    "Logistic": Pipeline([
        (
            "scale",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])
}


# ============================================================
# Feature selection
# IMPORTANT:
# Selection happens ONLY inside training fold
# ============================================================

def select_genes(
    X_train,
    y_train,
    n_genes=15
):

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    importance = pd.DataFrame({

        "Gene": X_train.columns,

        "Importance":
        rf.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    return importance.head(
        n_genes
    )["Gene"].tolist()


# ============================================================
# Store results
# ============================================================

all_results = []


# ============================================================
# Store selected genes
# ============================================================

gene_records = []


# ============================================================
# OUTER LOOP
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y),
    start=1
):

    print("\n==========================================")
    print("OUTER FOLD:", fold)
    print("==========================================")

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )


    # --------------------------------------------------------
    # Feature selection
    # ONLY training data
    # --------------------------------------------------------

    genes = select_genes(
        X_train,
        y_train,
        n_genes=15
    )


    print("\nSelected genes:")

    for i, gene in enumerate(
        genes,
        start=1
    ):

        print(
            f"{i:02d}. {gene}"
        )


    # Save genes for this fold

    gene_records.append({

        "Fold": fold,

        "Genes":
        ";".join(genes)

    })


    # --------------------------------------------------------
    # Apply selected genes
    # --------------------------------------------------------

    X_train_selected = X_train[
        genes
    ]

    X_test_selected = X_test[
        genes
    ]


    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for name, model in models.items():

        print(
            "\nRunning:",
            name
        )


        model.fit(
            X_train_selected,
            y_train
        )


        # ----------------------------------------------------
        # Predict ONLY unseen test fold
        # ----------------------------------------------------

        prob = model.predict_proba(
            X_test_selected
        )[:, 1]


        pred = (
            prob >= 0.5
        ).astype(int)


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        auc = roc_auc_score(
            y_test,
            prob
        )


        accuracy = accuracy_score(
            y_test,
            pred
        )


        cm = confusion_matrix(
            y_test,
            pred
        )


        print(
            "AUC:",
            round(auc, 4)
        )


        print(
            "Accuracy:",
            round(accuracy, 4)
        )


        print(
            "Confusion Matrix:"
        )

        print(cm)


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append({

            "Fold": fold,

            "Model": name,

            "AUC": auc,

            "Accuracy": accuracy,

            "Genes":
            ";".join(genes)

        })


# ============================================================
# Fold results
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n\n==========================================")
print("FOLD RESULTS")
print("==========================================\n")


print(results_df)


# ============================================================
# Summary
# ============================================================

summary = (

    results_df
    .groupby("Model")
    .agg(

        AUC_mean=("AUC", "mean"),

        AUC_std=("AUC", "std"),

        Accuracy_mean=("Accuracy", "mean"),

        Accuracy_std=("Accuracy", "std")

    )
    .reset_index()

)


print("\n==========================================")
print("NESTED CV SUMMARY")
print("==========================================\n")


print(summary)


# ============================================================
# Gene stability across outer folds
# ============================================================

gene_df = pd.DataFrame(
    gene_records
)


gene_counts = {}

for genes_string in gene_df["Genes"]:

    genes = genes_string.split(";")

    for gene in genes:

        if gene not in gene_counts:
            gene_counts[gene] = 0

        gene_counts[gene] += 1


gene_stability = pd.DataFrame({

    "Gene":
    list(gene_counts.keys()),

    "Selected_Folds":
    list(gene_counts.values())

})


gene_stability[
    "Selection_%"
] = (

    gene_stability[
        "Selected_Folds"
    ]

    / 5

    * 100

)


gene_stability = gene_stability.sort_values(
    "Selection_%",
    ascending=False
)


print("\n==========================================")
print("GENE STABILITY ACROSS OUTER FOLDS")
print("==========================================\n")


print(
    gene_stability
)


# ============================================================
# Save results
# ============================================================

results_df.to_csv(
    "data/nested_cv_fold_results.csv",
    index=False
)


summary.to_csv(
    "data/nested_cv_summary.csv",
    index=False
)


gene_df.to_csv(
    "data/nested_cv_selected_genes.csv",
    index=False
)


gene_stability.to_csv(
    "data/nested_cv_gene_stability.csv",
    index=False
)


print("\n==========================================")
print("FINISHED")
print("==========================================")


print("\nSaved:")

print(
    "data/nested_cv_fold_results.csv"
)

print(
    "data/nested_cv_summary.csv"
)

print(
    "data/nested_cv_selected_genes.csv"
)

print(
    "data/nested_cv_gene_stability.csv"
)