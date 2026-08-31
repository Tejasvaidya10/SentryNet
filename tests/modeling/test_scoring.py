import numpy as np
import pandas as pd

from sentrynet.modeling.scoring import score_transaction


class FakeModel:
    def __init__(self):
        self.received_features = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.received_features = X
        return np.array([[0.2, 0.8]])


def test_score_transaction_uses_looked_up_entity_stats():
    model = FakeModel()
    transaction = {"TransactionID": 1, "TransactionAmt": 100.0, "device": "d1"}
    entity_store = {"device:d1": {"degree": 5, "component_size": 3, "community_id": 2, "clustering_coeff": 0.4}}

    result = score_transaction(
        transaction,
        model=model,
        feature_columns=["TransactionAmt", "device_degree", "is_new_entity"],
        entity_lookup=lambda key: entity_store.get(key),
        entity_key_builders={"device": lambda t: f"device:{t['device']}"},
    )

    assert result.transaction_id == 1
    assert result.fraud_probability == 0.8
    assert result.is_new_entity is False
    assert model.received_features.iloc[0]["device_degree"] == 5
    assert model.received_features.iloc[0]["is_new_entity"] == 0


def test_score_transaction_cold_start_uses_fallback_and_flags_new_entity():
    model = FakeModel()
    transaction = {"TransactionID": 2, "TransactionAmt": 50.0, "device": "unseen"}

    result = score_transaction(
        transaction,
        model=model,
        feature_columns=["TransactionAmt", "device_degree", "is_new_entity"],
        entity_lookup=lambda key: None,
        entity_key_builders={"device": lambda t: f"device:{t['device']}"},
    )

    assert result.is_new_entity is True
    assert model.received_features.iloc[0]["device_degree"] == 0
    assert model.received_features.iloc[0]["is_new_entity"] == 1
