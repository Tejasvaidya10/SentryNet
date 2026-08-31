import pandas as pd

from sentrynet.graph.build import build_bipartite_graph
from sentrynet.graph.features import extract_entity_features


def test_extract_entity_features_degree_and_component_size():
    # 3 transactions share device "d1" (degree 3); 1 transaction has its own device "d2".
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "device": ["d1", "d1", "d1", "d2"],
        }
    )
    g = build_bipartite_graph(df, transaction_id_col="TransactionID", entity_cols=("device",))
    result = extract_entity_features(g)

    d1_key = ("entity", "device", "d1")
    d2_key = ("entity", "device", "d2")

    def row_for(key):
        return result[result["entity_key"] == key].iloc[0]

    assert row_for(d1_key)["degree"] == 3
    assert row_for(d2_key)["degree"] == 1
    # d1's component contains txns 1,2,3 + entity d1 = 4 nodes; d2's component contains txn 4 + entity d2 = 2 nodes.
    assert row_for(d1_key)["component_size"] == 4
    assert row_for(d2_key)["component_size"] == 2


def test_extract_entity_features_handles_graph_with_no_shared_entities():
    # No entity is shared by more than one transaction -> no projected edges, no communities.
    df = pd.DataFrame({"TransactionID": [1, 2], "device": ["d1", "d2"]})
    g = build_bipartite_graph(df, transaction_id_col="TransactionID", entity_cols=("device",))
    result = extract_entity_features(g)
    assert (result["community_id"] == -1).all()
    assert (result["clustering_coeff"] == 0.0).all()
