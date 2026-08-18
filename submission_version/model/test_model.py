import joblib
import pandas as pd

model = joblib.load("model/final_parkinson_logistic_model.pkl")

genes = open("model/final_genes.txt").read().splitlines()

coef = pd.DataFrame({
    "Gene": genes,
    "Coefficient": model.coef_[0]
})

coef = coef.sort_values("Coefficient", key=abs, ascending=False)

print(coef)