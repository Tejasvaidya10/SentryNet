from dataclasses import dataclass
from typing import Callable

import pandas as pd

DEFAULT_ENTITY_FEATURE_FALLBACK = {
    "degree": 0,
    "component_size": 1,
    "community_id": -1,
    "clustering_coeff": 0.0,
}


@dataclass
class ScoredTransaction:
    transaction_id: object
    fraud_probability: float
    is_new_entity: bool


def score_transaction(
    transaction: dict,
    model,
    feature_columns: list[str],
    entity_lookup: Callable[[str], dict | None],
    entity_key_builders: dict[str, Callable[[dict], str]],
) -> ScoredTransaction:
    """Score one transaction via precomputed entity-store lookups.

    No graph traversal happens here: entity_lookup is expected to be a
    point lookup (e.g. a Redis GET) against a table that a separate
    batch/streaming job keeps up to date. See docs/production_scaling.md.
    """
    row = dict(transaction)
    is_new_entity = False
    for prefix, key_builder in entity_key_builders.items():
        key = key_builder(transaction)
        stats = entity_lookup(key)
        if stats is None:
            stats = DEFAULT_ENTITY_FEATURE_FALLBACK
            is_new_entity = True
        for stat_name, value in stats.items():
            row[f"{prefix}_{stat_name}"] = value
    row["is_new_entity"] = int(is_new_entity)

    features = pd.DataFrame([{col: row.get(col) for col in feature_columns}])
    probability = float(model.predict_proba(features)[0, 1])

    return ScoredTransaction(
        transaction_id=transaction.get("TransactionID"),
        fraud_probability=probability,
        is_new_entity=is_new_entity,
    )
