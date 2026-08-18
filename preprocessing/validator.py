from pathlib import Path
import pandas as pd

from preprocessing.model_loader import load_model


class InputValidator:

    def __init__(self):

        _, _, self.required_genes = load_model()

    def validate(self, csv_file):

        csv_file = Path(csv_file)

        if not csv_file.exists():
            raise FileNotFoundError(
                f"{csv_file} not found."
            )

        df = pd.read_csv(csv_file)

        missing = []

        for gene in self.required_genes:

            if gene not in df.columns:
                missing.append(gene)

        if missing:

            raise ValueError(
                f"Missing genes: {missing}"
            )

        return True


if __name__ == "__main__":

    print("="*50)
    print("Validator Ready")
    print("="*50)