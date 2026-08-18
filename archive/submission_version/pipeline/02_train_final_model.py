"""
Train مدل نهایی روی GSE99039 با لیست ژن قفل‌شده (gene_signature_locked.json)

رفع مشکلات نسخه‌ی قبلی:
- "Training AUC" حذف شد (چون معیار معتبری برای تعمیم‌پذیری نیست)
- به‌جایش: nested-style 10-fold Stratified CV گزارش می‌شود که واقعی‌ترین
  تخمین از عملکرد مدل روی داده‌ی دیده‌نشده است
- Scaler و مدل نهایی (روی کل داده) جداگانه ذخیره می‌شوند، اما عدد AUC
  گزارش‌شده همیشه از CV می‌آید نه از fit روی همان داده

ورودی:  data/GSE99039_final_dataset.csv, data/GSE99039_labels.csv,
        gene_signature_locked.json
خروجی:  model/final_parkinson_logistic_model.pkl
        model/final_scaler.pkl
        model/final_genes.txt
        model/cv_performance.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

DATA_EXPR = "data/GSE99039_final_dataset.csv"
DATA_LABELS = "data/GSE99039_labels.csv"
GENE_SIGNATURE = "gene_signature_locked.json"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("TRAIN FINAL PARKINSON MODEL (fixed pipeline)")
    print("=" * 60)

    with open(GENE_SIGNATURE, encoding="utf-8") as f:
        signature = json.load(f)
    genes = signature["genes"]
    print("Locked gene signature version:", signature["version"])
    print("Genes (%d):" % len(genes), genes)

    expression = pd.read_csv(DATA_EXPR)
    labels = pd.read_csv(DATA_LABELS)

    if "Sample" not in expression.columns:
        expression = expression.rename(columns={expression.columns[0]: "Sample"})

    data = expression.merge(labels, on="Sample")
    print("\nMerged dataset:", data.shape)

    missing = [g for g in genes if g not in data.columns]
    if missing:
        raise ValueError(
            f"این ژن‌ها در GSE99039_final_dataset.csv پیدا نشدند: {missing}. "
            "لیست gene_signature_locked.json را بررسی کنید."
        )

    X = data[genes].copy()
    y = data["Label"]

    print("\nClasses:")
    print(y.value_counts())

    # ==========================
    # نکته: اگر GSE99039_gene_expression.csv از قبل log2 از GEO آمده،
    # این خط را کامنت کنید تا دوبار log گرفته نشود. برای بررسی، توزیع
    # X.describe() را قبل و بعد از log مقایسه کنید (بازه‌ی منطقی 0-16).
    # ==========================
    if X.max().max() > 50:  # هشدار ساده: اگر مقادیر بزرگ بودند، احتمالاً خام‌اند
        print("\n[INFO] مقادیر بزرگ‌تر از ۵۰ دیده شد -> اعمال log2(x+1)")
        X = np.log2(X + 1)
    else:
        print(
            "\n[WARNING] مقادیر در بازه‌ی معمول log2 هستند. "
            "احتمالاً داده از قبل log-transformed است؛ log دوباره زده نشد. "
            "دستی بررسی کنید که این فرض درست باشد."
        )

    # ==========================
    # Cross-validated performance (تخمین واقعی، نه training-only)
    # ==========================
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scaler_cv = StandardScaler()
    X_scaled_for_cv = scaler_cv.fit_transform(X)

    model_cv = LogisticRegression(max_iter=5000, random_state=42)
    prob_cv = cross_val_predict(
        model_cv, X_scaled_for_cv, y, cv=cv, method="predict_proba"
    )[:, 1]
    pred_cv = (prob_cv >= 0.5).astype(int)

    cv_auc = roc_auc_score(y, prob_cv)
    cv_acc = accuracy_score(y, pred_cv)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATED PERFORMANCE (این عدد را در مقاله گزارش کنید)")
    print("=" * 60)
    print("CV AUC:", round(cv_auc, 4))
    print("CV Accuracy:", round(cv_acc, 4))
    print("\nConfusion Matrix (CV):")
    print(confusion_matrix(y, pred_cv))
    print("\nClassification Report (CV):")
    print(classification_report(y, pred_cv))

    # ==========================
    # مدل نهایی: روی کل داده fit می‌شود تا برای external validation
    # استفاده شود - اما AUC این مرحله در مقاله گزارش *نمی‌شود*،
    # فقط برای تولید artifact نهایی است
    # ==========================
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X)

    final_model = LogisticRegression(max_iter=5000, random_state=42)
    final_model.fit(X_final_scaled, y)

    joblib.dump(final_model, os.path.join(MODEL_DIR, "final_parkinson_logistic_model.pkl"))
    joblib.dump(final_scaler, os.path.join(MODEL_DIR, "final_scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "final_genes.txt"), "w", encoding="utf-8") as f:
        for g in genes:
            f.write(g + "\n")

    cv_performance = {
        "cv_auc": cv_auc,
        "cv_accuracy": cv_acc,
        "n_splits": 10,
        "gene_signature_version": signature["version"],
        "n_genes": len(genes),
        "n_samples": int(X.shape[0]),
    }
    with open(os.path.join(MODEL_DIR, "cv_performance.json"), "w", encoding="utf-8") as f:
        json.dump(cv_performance, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", os.path.join(MODEL_DIR, "final_parkinson_logistic_model.pkl"))
    print(" -", os.path.join(MODEL_DIR, "final_scaler.pkl"))
    print(" -", os.path.join(MODEL_DIR, "final_genes.txt"))
    print(" -", os.path.join(MODEL_DIR, "cv_performance.json"))


if __name__ == "__main__":
    main()
