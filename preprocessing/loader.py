import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_gse99039():
    df = pd.read_csv(DATA_DIR / "GSE99039_final_dataset.csv")

    X = df.drop(columns=["Disease", "Unnamed: 0"])
    y = df["Disease"]

    return X, y


if __name__ == "__main__":
    X, y = load_gse99039()

    print("=" * 50)
    print("GSE99039 Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Samples : {len(X)}")
    print(f"Genes   : {X.shape[1]}")
    print("\nClasses:")
    print(y.value_counts())

    print("\nFirst 10 genes:")
    print(list(X.columns[:10]))