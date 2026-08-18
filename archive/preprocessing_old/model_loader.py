from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


def load_model():

    model = joblib.load(
        MODEL_DIR / "final_parkinson_logistic_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "final_scaler.pkl"
    )

    with open(
        MODEL_DIR / "final_genes.txt",
        "r"
    ) as f:

        genes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return model, scaler, genes


if __name__ == "__main__":

    model, scaler, genes = load_model()

    print("=" * 50)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 50)

    print("Model :", type(model))
    print("Scaler:", type(scaler))
    print("Genes :", len(genes))

    print("\nGene List:")
    for gene in genes:
        print(gene)