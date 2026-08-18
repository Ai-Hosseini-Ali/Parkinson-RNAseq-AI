"""
نرمال‌سازی صحیح RNA-seq برای GSE165082
رفع مشکل: نسخه‌های قبلی مستقیماً log2(raw_count+1) می‌زدند
بدون نرمال‌سازی عمق کتابخانه (library size) -> این نسخه ابتدا CPM
سپس log2 می‌کند.

ورودی:  data/GSE165082_PD-CC.counts.txt
خروجی:  data/GSE165082_normalized_log_cpm.csv   (samples x genes)
        data/GSE165082_labels.csv
"""

import pandas as pd
import numpy as np
import mygene

RAW_COUNTS_PATH = "data/GSE165082_PD-CC.counts.txt"
OUT_EXPR_PATH = "data/GSE165082_normalized_log_cpm.csv"
OUT_LABELS_PATH = "data/GSE165082_labels.csv"


def main():
    print("=" * 60)
    print("STEP 1: Load raw counts")
    print("=" * 60)

    df = pd.read_csv(RAW_COUNTS_PATH, sep="\t", index_col=0)
    print("Raw shape (genes x samples):", df.shape)

    print("\n" + "=" * 60)
    print("STEP 2: Filter low-expression genes")
    print("=" * 60)
    # فیلتر روی مجموع خام قبل از هر نرمال‌سازی - این بخش درست بود در نسخه‌های قبلی
    filtered = df[df.sum(axis=1) >= 10]
    print("After filtering:", filtered.shape)

    print("\n" + "=" * 60)
    print("STEP 3: CPM normalization (per-sample library size)")
    print("=" * 60)
    # نکته‌ی کلیدی که در نسخه‌های قبلی جا افتاده بود:
    # اندازه‌ی کتابخانه باید روی داده‌ی فیلترشده و به‌ازای هر ستون (نمونه) محاسبه شود
    library_size = filtered.sum(axis=0)
    print("Library sizes (first 5 samples):")
    print(library_size.head())

    cpm = filtered.div(library_size, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    print("\n" + "=" * 60)
    print("STEP 4: Map Ensembl IDs -> Gene Symbols")
    print("=" * 60)
    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        log_cpm.index.tolist(),
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
    )
    mapping = {r["query"]: r["symbol"] for r in result if "symbol" in r}
    print("Mapped:", len(mapping), "/", len(log_cpm.index))

    log_cpm["GeneSymbol"] = [mapping.get(g, None) for g in log_cpm.index]
    log_cpm = log_cpm.dropna(subset=["GeneSymbol"])
    log_cpm = log_cpm.set_index("GeneSymbol")
    # اگر چند Ensembl ID به یک symbol نگاشت شدند، میانگین بگیرید (نه جمع -
    # چون این‌ها دیگر log-CPM هستند، جمع کردن مقادیر log بی‌معنی است)
    log_cpm = log_cpm.groupby(log_cpm.index).mean()

    print("Final gene x sample matrix:", log_cpm.shape)

    print("\n" + "=" * 60)
    print("STEP 5: Build labels from sample names")
    print("=" * 60)
    labels = []
    for s in log_cpm.columns:
        if "PD" in s:
            labels.append(1)
        elif "CC" in s:
            labels.append(0)
        else:
            labels.append(np.nan)

    labels_df = pd.DataFrame({"Sample": log_cpm.columns, "Label": labels})
    labels_df = labels_df.dropna(subset=["Label"])
    labels_df["Label"] = labels_df["Label"].astype(int)

    valid_samples = labels_df["Sample"].tolist()
    log_cpm = log_cpm[valid_samples]

    print("Classes:")
    print(labels_df["Label"].value_counts())

    print("\n" + "=" * 60)
    print("SAVE")
    print("=" * 60)
    # transpose به samples x genes برای سازگاری با بقیه pipeline
    log_cpm.T.to_csv(OUT_EXPR_PATH)
    labels_df.to_csv(OUT_LABELS_PATH, index=False)

    print("Saved:", OUT_EXPR_PATH)
    print("Saved:", OUT_LABELS_PATH)


if __name__ == "__main__":
    main()
