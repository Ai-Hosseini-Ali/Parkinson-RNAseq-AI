import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# SHAP NESTED / LEAKAGE-CONTROLLED ANALYSIS
# GSE99039
# ============================================================

print("\n==========================================")
print("SHAP NESTED / LEAKAGE-CONTROLLED ANALYSIS")
print("GSE99039")
print("==========================================\n")


# ============================================================
# Load dataset
# ============================================================

data = pd.read_csv(
    "data/GSE99039_top100_dataset.csv"
)

print("Dataset:")
print(data.shape)


# ============================================================
# Separate X / y
# ============================================================

X = data.drop(
    columns=["Disease"]
)

y = data["Disease"]


print("\nNumber of genes:")
print(X.shape[1])


print("\nClasses:")
print(y.value_counts())


# ============================================================
# Outer CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# Storage
# ============================================================

shap_results = []

gene_selection_results = []


# ============================================================
# Models
# ============================================================

models = {

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
    ]),

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
    ])
}


# ============================================================
# Feature selection
# ============================================================

from sklearn.ensemble import RandomForestClassifier


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

        "Gene":
        X_train.columns,

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
# Outer loop
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    outer_cv.split(X, y),
    start=1
):

    print("\n==========================================")
    print("OUTER FOLD:", fold)
    print("==========================================")

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


    gene_selection_results.append({

        "Fold":
        fold,

        "Genes":
        ";".join(genes)

    })


    X_train_selected = X_train[
        genes
    ]

    X_test_selected = X_test[
        genes
    ]


    # ========================================================
    # SHAP for each model
    # ========================================================

    for model_name, model in models.items():

        print(
            f"\nRunning SHAP:",
            model_name
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train_selected,
            y_train
        )


        # ----------------------------------------------------
        # Transform data
        # ----------------------------------------------------

        scaler = model.named_steps[
            "scale"
        ]

        trained_model = model.named_steps[
            "model"
        ]


        X_train_scaled = scaler.transform(
            X_train_selected
        )

        X_test_scaled = scaler.transform(
            X_test_selected
        )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        if model_name == "Logistic":

            explainer = shap.LinearExplainer(
                trained_model,
                X_train_scaled
            )

            shap_values = explainer.shap_values(
                X_test_scaled
            )


        elif model_name == "SVM":

            explainer = shap.LinearExplainer(
                trained_model,
                X_train_scaled
            )

            shap_values = explainer.shap_values(
                X_test_scaled
            )


        # ----------------------------------------------------
        # Absolute SHAP
        # ----------------------------------------------------

        mean_abs_shap = np.abs(
            shap_values
        ).mean(
            axis=0
        )


        fold_shap = pd.DataFrame({

            "Gene":
            genes,

            "Mean_Abs_SHAP":
            mean_abs_shap,

            "Fold":
            fold,

            "Model":
            model_name

        })


        shap_results.append(
            fold_shap
        )


        print("\nTop SHAP genes:")

        print(
            fold_shap
            .sort_values(
                "Mean_Abs_SHAP",
                ascending=False
            )
            .head(15)
        )


# ============================================================
# Combine
# ============================================================

shap_df = pd.concat(
    shap_results,
    ignore_index=True
)


# ============================================================
# SHAP summary across folds
# ============================================================

shap_summary = (

    shap_df

    .groupby(
        ["Gene", "Model"]
    )

    .agg(

        Mean_SHAP=(
            "Mean_Abs_SHAP",
            "mean"
        ),

        Std_SHAP=(
            "Mean_Abs_SHAP",
            "std"
        ),

        Folds=(
            "Fold",
            "count"
        )

    )

    .reset_index()

)


shap_summary = shap_summary.sort_values(
    "Mean_SHAP",
    ascending=False
)


# ============================================================
# Save
# ============================================================

shap_df.to_csv(
    "data/shap_nested_fold_results.csv",
    index=False
)


shap_summary.to_csv(
    "data/shap_nested_summary.csv",
    index=False
)


pd.DataFrame(
    gene_selection_results
).to_csv(
    "data/shap_nested_selected_genes.csv",
    index=False
)


# ============================================================
# Final output
# ============================================================

print("\n\n==========================================")
print("SHAP SUMMARY")
print("==========================================\n")


print(
    shap_summary.head(30)
)


print("\n==========================================")
print("FINISHED")
print("==========================================")

print("\nSaved:")

print(
    "data/shap_nested_fold_results.csv"
)

print(
    "data/shap_nested_summary.csv"
)

print(
    "data/shap_nested_selected_genes.csv"
)