import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


GENES = [
    'PTGDS',
    'KIR2DL1',
    'KIR2DL3',
    'LILRB1',
    'KIAA0319L',
    'PPP4C',
    'TYROBP',
    'MBOAT7',
    'MMP9',
    'HLA-C',
    'LRRC25',
    'KIR3DL1',
    'BCL2',
    'RHOG'
]


BASE = r"C:\Users\alihoseini\Desktop\Parkinson_AI\data"


train = pd.read_csv(
    BASE + r"\GSE99039_final_dataset.csv",
    index_col=0
)


labels = pd.read_csv(
    BASE + r"\GSE99039_labels.csv"
)


ext = pd.read_csv(
    BASE + r"\GSE165082_normalized_log_cpm.csv",
    index_col=0
)


ext_labels = pd.read_csv(
    BASE + r"\GSE165082_labels.csv"
)



X_train = train[GENES]

X_ext = ext[GENES]


y_train = labels["Label"]



scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

X_ext = scaler.transform(X_ext)



model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train,
    y_train
)



probs = model.predict_proba(X_ext)[:,1]


result = pd.DataFrame({

    "Sample": ext.index,

    "Probability_PD": probs,

    "True_Label": ext_labels["Label"]

})


print(result)


result.to_csv(
    "results/GSE165082_probabilities.csv",
    index=False
)


print("\nSaved:")
print("results/GSE165082_probabilities.csv")