import joblib
import os

path = os.path.join(
    os.path.dirname(__file__),
    "final_parkinson_logistic_model.pkl"
)

print("Loading:", path)

model = joblib.load(path)

print("\nModel type:")
print(type(model))

print("\nModel parameters:")
print(model.get_params())

print("\nNumber of features:")
print(model.n_features_in_)

print("\nClasses:")
print(model.classes_)

print("\nCoefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)