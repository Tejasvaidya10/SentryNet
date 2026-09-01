import networkx as nx
import pandas as pd


def build_bipartite_graph(
    df: pd.DataFrame, transaction_id_col: str, entity_cols: tuple[str, ...]
) -> nx.Graph:
    """Build a transaction<->entity bipartite graph.

    A NaN entity value means "no signal" (e.g. no identity data at all for
    that transaction), not a shared identity -- it is skipped rather than
    turned into an edge, since otherwise every transaction lacking that
    signal would collide into one giant supernode and make the later
    projection step (see extract_entity_features) computationally
    intractable.
    """
    g = nx.Graph()
    columns = [transaction_id_col, *entity_cols]
    for row in df[columns].itertuples(index=False):
        txn_node = ("txn", getattr(row, transaction_id_col))
        g.add_node(txn_node, bipartite=0)
        for col in entity_cols:
            value = getattr(row, col)
            if pd.isna(value):
                continue
            entity_node = ("entity", col, value)
            g.add_node(entity_node, bipartite=1)
            g.add_edge(txn_node, entity_node)
    return g
