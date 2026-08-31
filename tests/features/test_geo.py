from sentrynet.features.geo import addr_change_flag
import pandas as pd


def test_addr_change_flag_true_only_when_addr_changes():
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "A"],
            "TransactionDT": [0, 100, 200],
            "addr1": [100, 100, 200],
            "addr2": [10, 10, 10],
        }
    )
    result = addr_change_flag(df, entity_col="entity", time_col="TransactionDT")
    assert result.tolist() == [False, False, True]


def test_addr_change_flag_false_for_first_transaction_per_entity():
    df = pd.DataFrame(
        {
            "entity": ["A", "B"],
            "TransactionDT": [0, 0],
            "addr1": [100, 200],
            "addr2": [10, 20],
        }
    )
    result = addr_change_flag(df, entity_col="entity", time_col="TransactionDT")
    assert result.tolist() == [False, False]
