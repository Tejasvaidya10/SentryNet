import pandas as pd

from sentrynet.graph.build import build_bipartite_graph


def test_build_bipartite_graph_connects_transactions_to_shared_entities():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "device": ["d1", "d1", "d2"],
        }
    )
    g = build_bipartite_graph(df, transaction_id_col="TransactionID", entity_cols=("device",))

    txn1, txn2, txn3 = ("txn", 1), ("txn", 2), ("txn", 3)
    entity_d1 = ("entity", "device", "d1")
    entity_d2 = ("entity", "device", "d2")

    assert g.has_edge(txn1, entity_d1)
    assert g.has_edge(txn2, entity_d1)
    assert g.has_edge(txn3, entity_d2)
    assert not g.has_edge(txn3, entity_d1)
    # Transactions sharing entity d1 are connected via that entity node.
    assert entity_d1 in g[txn1] and entity_d1 in g[txn2]
