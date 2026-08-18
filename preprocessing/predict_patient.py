import pandas as pd
import numpy as np
import joblib


print("==============================")
print("PARKINSON AI PREDICTOR")
print("==============================")


# ==========================
# Load model
# ==========================

model = joblib.load(
    "data/final_parkinson_logistic_model.pkl"
)


scaler = joblib.load(
    "data/final_scaler.pkl"
)



# ==========================
# Load genes
# ==========================

with open(
    "data/final_genes.txt",
    "r"
) as f:

    genes = [
        x.strip()
        for x in f.readlines()
    ]



print("\nRequired genes:")
print(genes)



# ==========================
# Input patient
# ==========================

print("\nEnter gene expression values")


patient = {}


for g in genes:

    while True:

        try:

            value = float(
                input(
                    f"{g}: "
                )
            )

            patient[g]=value

            break

        except:

            print(
                "Please enter number"
            )



# ==========================
# DataFrame
# ==========================

X = pd.DataFrame(
    [patient]
)



# same preprocessing

X = np.log2(
    X + 1
)



# scaling

X_scaled = scaler.transform(
    X
)



# ==========================
# Prediction
# ==========================


prob = model.predict_proba(
    X_scaled
)[0,1]



prediction = model.predict(
    X_scaled
)[0]



print("\n===================")
print("RESULT")
print("===================")


print(
f"Parkinson probability: {prob*100:.2f}%"
)



if prediction == 1:

    print(
        "Prediction: Parkinson / IPD"
    )

else:

    print(
        "Prediction: Control"
    )


print("===================")