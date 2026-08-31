import pandas as pd

from sentrynet.data.split import temporal_split


def test_temporal_split_has_no_leakage():
    df = pd.DataFrame({"TransactionDT": list(range(100)), "value": list(range(100))})
    train, test = temporal_split(df, time_col="TransactionDT", train_frac=0.7)
    assert len(train) + len(test) == len(df)
    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) > 0 and len(test) > 0


def test_temporal_split_respects_train_frac_roughly():
    df = pd.DataFrame({"TransactionDT": list(range(100)), "value": list(range(100))})
    train, test = temporal_split(df, time_col="TransactionDT", train_frac=0.7)
    assert 65 <= len(train) <= 75
