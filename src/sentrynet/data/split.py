import pandas as pd


def temporal_split(
    df: pd.DataFrame, time_col: str = "TransactionDT", train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df[time_col].quantile(train_frac)
    train = df[df[time_col] <= cutoff]
    test = df[df[time_col] > cutoff]
    return train, test
