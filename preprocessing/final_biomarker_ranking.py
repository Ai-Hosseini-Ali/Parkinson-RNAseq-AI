import pandas as pd
import networkx as nx


# -------------------------
# Load ML importance
# -------------------------

ml = pd.read_csv(
    "data/GSE99039_feature_importance.csv"
)

print("ML:")
print(ml.head())


# -------------------------
# Load DE results
# -------------------------

de = pd.read_csv(
    "data/GSE99039_DE_FDR.csv"
)

print("DE:")
print(de.head())


# -------------------------
# Load STRING network
# -------------------------

ppi = pd.read_csv(
    "data/string_network_edges.csv"
)

# Build network
G = nx.Graph()

for _, row in ppi.iterrows():

    G.add_edge(
        row["preferredName_A"],
        row["preferredName_B"]
    )


centrality = nx.degree_centrality(G)


ppi_score = pd.DataFrame(
    {
        "Gene": list(centrality.keys()),
        "PPI_score": list(centrality.values())
    }
)


# -------------------------
# Normalize scores
# -------------------------

def normalize(x):

    return (
        (x-x.min()) /
        (x.max()-x.min())
    )


ml["ML_score"] = normalize(
    ml["Importance"]
)


de["DE_score"] = normalize(
    -de["FDR"].apply(lambda x: x)
)


ppi_score["PPI_score"] = normalize(
    ppi_score["PPI_score"]
)



# -------------------------
# Merge
# -------------------------

result = (
    ml[["Gene","ML_score"]]
    .merge(
        de[["Gene","DE_score"]],
        on="Gene",
        how="inner"
    )
    .merge(
        ppi_score,
        on="Gene",
        how="inner"
    )
)


print("Merged:")
print(result.shape)


# -------------------------
# Final score
# -------------------------

result["Final_score"] = (
    0.4*result["ML_score"]
    +
    0.3*result["DE_score"]
    +
    0.3*result["PPI_score"]
)


result = result.sort_values(
    "Final_score",
    ascending=False
)


print("\nTOP BIOMARKERS")
print(
    result.head(30)
)


result.to_csv(
    "data/final_ranked_biomarkers.csv",
    index=False
)


print("\nSaved")