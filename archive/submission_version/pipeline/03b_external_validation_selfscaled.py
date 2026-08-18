"""
External validation on GSE165082 - self-scaled version
Difference from 03_external_validation.py:
  Instead of applying the StandardScaler fitted on GSE99039 (training
  cohort) to GSE165082, we fit an INDEPENDENT StandardScaler on
  GSE165082 itself (only on the 14 locked signature genes), using no
  label information. This addresses cross-platform scale mismatch
  without touching the locked model coefficients or using test labels
  in any way that could leak into model fitting.
Input:  data/GSE165082_normalized_log_cpm.csv (output of step 01)
        data/GSE165082_labels.csv (output of step 01)
        model/final_parkinson_logistic_model.pkl (output of step 02)
        gene_signature_locked.json
Output: results/GSE165082_external_validation_SELFSCALED.csv
"""
import json
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

EXT_EXPR = "data/GSE165082_normalized_log_cpm.csv"
EXT_LABELS = "data/GSE165082_labels.csv"
GENE_SIGNATURE = "gene_signature_locked.json"
MODEL_PATH = "model/final_parkinson_logistic_model.pkl"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("EXTERNAL VALIDATION on GSE165082 (self-scaled)")
    print("=" * 60)

    with open(GENE_SIGNATURE, encoding="utf-8") as f:
        signature = json.load(f)
    genes = signature["genes"]
    print("Using locked gene signature:", signature["version"])

    print("\nLoading fitted model (trained on GSE99039)...")
    model = joblib.load(MODEL_PATH)

    print("Loading normalized external data (CPM + log2, from step 01)...")
    ext = pd.read_csv(EXT_EXPR, index_col=0)  # samples x genes
    labels_df = pd.read_csv(EXT_LABELS)

    missing = [g for g in genes if g not in ext.columns]
    if missing:
        raise ValueError(
            f"These genes were not found in the normalized external data: {missing}. "
            "Check the annotation/mapping step in step 01."
        )

    ext = ext.loc[labels_df["Sample"]]
    X_ext = ext[genes]
    y_ext = labels_df["Label"].values

    print("\nExternal set shape:", X_ext.shape)
    print("Classes:")
    print(pd.Series(y_ext).value_counts())

    # KEY CHANGE: fit a fresh scaler on GSE165082 itself, not the
    # scaler fitted on GSE99039. This is unsupervised (no labels used)
    # and only affects the scale of inputs into the locked model.
    print("\nFitting independent StandardScaler on GSE165082 (self-scaling)...")
    ext_scaler = StandardScaler()
    X_ext_scaled = ext_scaler.fit_transform(X_ext)

    prob = model.predict_proba(X_ext_scaled)[:, 1]
    pred = model.predict(X_ext_scaled)

    auc = roc_auc_score(y_ext, prob)
    acc = accuracy_score(y_ext, pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("AUC:", round(auc, 4))
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_ext, pred))
    print("\nClassification Report:")
    print(classification_report(y_ext, pred, zero_division=0))

    if len(set(pred)) == 1:
        print(
            "\n[WARNING] Model still predicts a single class for all samples. "
            "Self-scaling alone was not sufficient; a more thorough "
            "harmonization method (e.g. ComBat) is likely needed."
        )

    result = pd.DataFrame(
        {
            "Sample": X_ext.index,
            "True_Label": y_ext,
            "Prediction": pred,
            "PD_probability": prob,
        }
    )
    out_path = os.path.join(RESULTS_DIR, "GSE165082_external_validation_SELFSCALED.csv")
    result.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
