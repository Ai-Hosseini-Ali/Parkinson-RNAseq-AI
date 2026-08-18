import pandas as pd


def detect_gene_column(df):

    possible_columns = [
        "Gene",
        "GeneSymbol",
        "gene",
        "symbol",
        "GeneID"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None



def extract_signature(df, signature):

    gene_column = detect_gene_column(df)

    if gene_column is None:
        raise Exception(
            "Gene column not found"
        )


    df[gene_column] = df[gene_column].astype(str)


    available = [
        g for g in signature
        if g in df[gene_column].values
    ]


    missing = set(signature) - set(available)


    print("================")
    print("Available genes:")
    print(available)

    print("================")
    print("Missing genes:")
    print(missing)


    result = df[
        df[gene_column].isin(available)
    ]


    return result