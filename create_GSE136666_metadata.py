import pandas as pd


samples = [
"GSM4054763_Counts_SN_C1",
"GSM4054764_Counts_SN_C2",
"GSM4054765_Counts_SN_C3",
"GSM4054766_Counts_SN_C4",
"GSM4054767_Counts_SN_C5",
"GSM4054768_Counts_SN_PD1",
"GSM4054769_Counts_SN_PD2",
"GSM4054770_Counts_SN_PD3",
"GSM4054771_Counts_SN_PD4",
"GSM4054772_Counts_SN_PD5"
]


condition = [
"Control",
"Control",
"Control",
"Control",
"Control",
"Parkinson",
"Parkinson",
"Parkinson",
"Parkinson",
"Parkinson"
]


metadata = pd.DataFrame({
    "Sample": samples,
    "Condition": condition
})


metadata.to_csv(
    "data/GEO/GSE136666/processed/GSE136666_metadata.csv",
    index=False
)


print(metadata)
print("\nSaved!")