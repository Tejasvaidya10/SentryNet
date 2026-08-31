import numpy as np
import pandas as pd

from sentrynet.features.recency import time_since_last


def test_time_since_last_is_nan_for_first_transaction():
    df = pd.DataFrame({"entity": ["A", "A"], "TransactionDT": [0, 100]})
    result = time_since_last(df, entity_col="entity", time_col="TransactionDT")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 100


def test_time_since_last_is_per_entity():
    df = pd.DataFrame({"entity": ["A", "B", "A"], "TransactionDT": [0, 0, 50]})
    result = time_since_last(df, entity_col="entity", time_col="TransactionDT")
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 50
