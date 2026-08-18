import pandas as pd
import networkx as nx
import requests
import matplotlib.pyplot as plt


# ==========================
# Load biomarkers
# ==========================

df = pd.read_csv(
    "data/pathway_input_biomarkers.csv"
)

print("Input:")
print(df.shape)

genes = df["Gene"].dropna().unique().tolist()

print("Number of genes:", len(genes))


# ==========================
# STRING API
# ==========================

print("\nQuerying STRING database...")


string_url = "https://string-db.org/api/tsv/network"


params = {
    "identifiers": "%0d".join(genes),
    "species": 9606,   # Human
    "required_score": 400,
    "network_flavor": "confidence",
    "caller_identity": "Parkinson_AI"
}


response = requests.get(
    string_url,
    params=params
)


if response.status_code != 200:
    raise Exception("STRING API failed")


# ==========================
# Parse interactions
# ==========================

from io import StringIO


ppi = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)


print("\nSTRING interactions:")
print(ppi.shape)

print(ppi.head())


# ==========================
# Save edges
# ==========================

ppi.to_csv(
    "data/string_network_edges.csv",
    index=False
)


# ==========================
# Build graph
# ==========================

G = nx.Graph()


for _, row in ppi.iterrows():

    G.add_edge(
        row["preferredName_A"],
        row["preferredName_B"],
        score=row["score"]
    )


print("\nNetwork:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ==========================
# Hub genes
# ==========================

degree = nx.degree_centrality(G)


hub = (
    pd.DataFrame(
        degree.items(),
        columns=["Gene","Degree"]
    )
    .sort_values(
        "Degree",
        ascending=False
    )
)


print("\nTop hub genes:")
print(hub.head(20))


hub.to_csv(
    "data/string_hub_genes.csv",
    index=False
)


# ==========================
# Plot network
# ==========================

plt.figure(figsize=(12,10))


top_nodes = hub.head(50)["Gene"].tolist()

sub = G.subgraph(top_nodes)


pos = nx.spring_layout(
    sub,
    seed=42
)


nx.draw(
    sub,
    pos,
    with_labels=True,
    node_size=500,
    font_size=8
)


plt.title(
    "STRING PPI Network - Parkinson Biomarkers"
)


plt.savefig(
    "data/string_network.png",
    dpi=300,
    bbox_inches="tight"
)


print("\nSaved successfully")