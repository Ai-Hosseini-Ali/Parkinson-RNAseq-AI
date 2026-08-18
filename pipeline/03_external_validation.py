"""
External validation روی GSE165082 - نسخه‌ی اصلاح‌شده

رفع مشکلات نسخه‌های قبلی:
1. باگ اصلی: قبلاً StandardScaler که روی میکروآرایه fit شده بود مستقیماً
   روی مقادیر خام RNA-seq اعمال می‌شد -> اشباع سیگموئید -> پیش‌بینی همه PD.
   این نسخه از داده‌ی از قبل نرمال‌شده (log-CPM از مرحله ۰۱) استفاده می‌کند.
2. مدل از نو train نمی‌شود؛ دقیقاً همان مدل قفل‌شده‌ی نهایی (از مرحله ۰۲)
   لود و اعمال می‌شود - نه یک لیست ژن یا مدل موازی متفاوت.
3. لیست ژن از gene_signature_locked.json خوانده می‌شود، نه hardcode جدا.

ورودی:  data/GSE165082_normalized_log_cpm.csv (خروجی مرحله ۰۱)
        data/GSE165082_labels.csv (خروجی مرحله ۰۱)
        model/final_parkinson_logistic_model.pkl (خروجی مرحله ۰۲)
        model/final_scaler.pkl
        gene_signature_locked.json
خروجی:  results/GSE165082_external_validation_FIXED.csv
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

EXT_EXPR = "data/GSE165082_normalized_log_cpm.csv"
EXT_LABELS = "data/GSE165082_labels.csv"
GENE_SIGNATURE = "gene_signature_locked.json"

MODEL_PATH = "model/final_parkinson_logistic_model.pkl"
SCALER_PATH = "model/final_scaler.pkl"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("EXTERNAL VALIDATION on GSE165082 (fixed pipeline)")
    print("=" * 60)

    with open(GENE_SIGNATURE, encoding="utf-8") as f:
        signature = json.load(f)
    genes = signature["genes"]
    print("Using locked gene signature:", signature["version"])

    print("\nLoading fitted model + scaler (trained on GSE99039)...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Loading normalized external data (CPM + log2, from step 01)...")
    ext = pd.read_csv(EXT_EXPR, index_col=0)  # samples x genes
    labels_df = pd.read_csv(EXT_LABELS)

    missing = [g for g in genes if g not in ext.columns]
    if missing:
        raise ValueError(
            f"این ژن‌ها در داده‌ی خارجی نرمال‌شده پیدا نشدند: {missing}. "
            "بررسی کنید annotate/mapping در مرحله ۰۱ آن‌ها را از دست نداده باشد."
        )

    ext = ext.loc[labels_df["Sample"]]
    X_ext = ext[genes]
    y_ext = labels_df["Label"].values

    print("\nExternal set shape:", X_ext.shape)
    print("Classes:")
    print(pd.Series(y_ext).value_counts())

    # همان scaler که روی GSE99039 fit شده - اما این‌بار روی داده‌ی
    # از قبل به‌درستی نرمال‌شده (log-CPM) اعمال می‌شود، نه مقادیر خام
    X_ext_scaled = scaler.transform(X_ext)

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
    print(classification_report(y_ext, pred))

    # هشدار خودکار اگر همچنان همه یک کلاس پیش‌بینی شد
    if len(set(pred)) == 1:
        print(
            "\n[WARNING] مدل همچنان همه‌ی نمونه‌ها را یک کلاس پیش‌بینی می‌کند. "
            "این یعنی مشکل احتمالاً عمیق‌تر از نرمال‌سازی است - به احتمال "
            "زیاد عدم سازگاری واقعی بین پلتفرم میکروآرایه و RNA-seq است "
            "و به یک روش harmonization جدی‌تر مثل ComBat نیاز دارید."
        )

    result = pd.DataFrame(
        {
            "Sample": X_ext.index,
            "True_Label": y_ext,
            "Prediction": pred,
            "PD_probability": prob,
        }
    )
    out_path = os.path.join(RESULTS_DIR, "GSE165082_external_validation_FIXED.csv")
    result.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
