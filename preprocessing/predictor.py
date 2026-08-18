from pathlib import Path

import pandas as pd

from preprocessing.model_loader import load_model


class ParkinsonPredictor:

    def __init__(self):

        self.model, self.scaler, self.genes = load_model()

    def prepare_sample(self, sample):

        if isinstance(sample, dict):

            sample = pd.DataFrame([sample])

        sample = sample[self.genes]

        sample = self.scaler.transform(sample)

        return sample

    def predict(self, sample):

        x = self.prepare_sample(sample)

        pred = self.model.predict(x)[0]

        prob = self.model.predict_proba(x)[0][1]

        return {

            "prediction": int(pred),

            "probability": float(prob)

        }


if __name__ == "__main__":

    predictor = ParkinsonPredictor()

    print("="*50)

    print("Predictor Ready")

    print("="*50)

    print()

    print("Expected genes:")

    for g in predictor.genes:

        print(g)