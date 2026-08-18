import joblib
import os

path = os.path.join(
    os.path.dirname(__file__),
    "final_scaler.pkl"
)

print("Loading:", path)

scaler = joblib.load(path)

print("\nScaler type:")
print(type(scaler))

print("\nScaler parameters:")
print(scaler.__dict__)