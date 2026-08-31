import networkx as nx
import pandas as pd


def build_bipartite_graph(
    df: pd.DataFrame, transaction_id_col: str, entity_cols: tuple[str, ...]
) -> nx.Graph:
    g = nx.Graph()
    for _, row in df.iterrows():
        txn_node = ("txn", row[transaction_id_col])
        g.add_node(txn_node, bipartite=0)
        for col in entity_cols:
            entity_node = ("entity", col, row[col])
            g.add_node(entity_node, bipartite=1)
            g.add_edge(txn_node, entity_node)
    return g
