import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# NESTED / LEAKAGE-CONTROLLED CROSS VALIDATION
# GSE99039
# ============================================================

print("\n==========================================")
print("NESTED / LEAKAGE-CONTROLLED VALIDATION")
print("GSE99039")
print("==========================================\n")


# ============================================================
# Paths
# ============================================================

expression_file = "data/GSE99039_final_dataset.csv"
labels_file = "data/GSE99039_labels.csv"


# ============================================================
# Load data
# ============================================================

expression = pd.read_csv(expression_file)
labels = pd.read_csv(labels_file)


# Fix sample column

if "Sample" not in expression.columns:

    expression = expression.rename(
        columns={
            expression.columns[0]: "Sample"
        }
    )


# Merge labels

data = expression.merge(
    labels[["Sample", "Label"]],
    on="Sample"
)


print("Dataset:")
print(data.shape)


# ============================================================
# Separate X / y
# ============================================================

drop_columns = [
    "Sample",
    "Label",
    "Disease"
]


feature_columns = [
    c for c in data.columns
    if c not in drop_columns
]


X = data[feature_columns].copy()
y = data["Label"].copy()


print("\nNumber of genes:")
print(len(feature_columns))


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
# Gene selection function
# ============================================================

def select_genes(X_train, y_train, n_genes=15):

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
# Store predictions
# ============================================================

all_results = []


# ============================================================
# Outer loop
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
    # Feature selection ONLY on training data
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


    # --------------------------------------------------------
    # Select same genes in train and test
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


        # Probability on unseen test fold

        prob = model.predict_proba(
            X_test_selected
        )[:, 1]


        pred = (
            prob >= 0.5
        ).astype(int)


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
        # Save fold results
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
# Results
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n\n==========================================")
print("FOLD RESULTS")
print("==========================================\n")

print(
    results_df
)


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


print(
    summary
)


# ============================================================
# Save
# ============================================================

results_df.to_csv(
    "data/nested_cv_fold_results.csv",
    index=False
)


summary.to_csv(
    "data/nested_cv_summary.csv",
    index=False
)


print("\n==========================================")
print("FINISHED")
print("==========================================")

print(
    "\nSaved:"
)

print(
    "data/nested_cv_fold_results.csv"
)

print(
    "data/nested_cv_summary.csv"
)