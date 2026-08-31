import pandas as pd

from sentrynet.features.velocity import transaction_velocity


def test_transaction_velocity_counts_prior_transactions_in_window():
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "A", "A"],
            "TransactionDT": [0, 100, 150, 500],
        }
    )
    result = transaction_velocity(df, entity_col="entity", time_col="TransactionDT", window_seconds=200)
    assert result.tolist() == [0, 1, 2, 0]


def test_transaction_velocity_is_per_entity():
    df = pd.DataFrame(
        {
            "entity": ["A", "B", "A", "B"],
            "TransactionDT": [0, 0, 50, 50],
        }
    )
    result = transaction_velocity(df, entity_col="entity", time_col="TransactionDT", window_seconds=200)
    assert result.tolist() == [0, 0, 1, 1]
