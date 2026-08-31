import pandas as pd

from sentrynet.graph.build import build_bipartite_graph
from sentrynet.graph.features import extract_entity_features
from sentrynet.graph.join import transaction_entity_features


def test_transaction_entity_features_joins_stats_by_membership():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "device": ["d1", "d1", "d2"],
        }
    )
    g = build_bipartite_graph(df, transaction_id_col="TransactionID", entity_cols=("device",))
    entity_features = extract_entity_features(g)

    result = transaction_entity_features(df, entity_cols=("device",), entity_features=entity_features)

    assert result["device_degree"].tolist() == [2, 2, 1]
    assert list(result.index) == list(df.index)
