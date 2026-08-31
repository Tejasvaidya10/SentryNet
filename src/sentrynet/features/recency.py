import pandas as pd


def time_since_last(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.Series:
    sorted_df = df.sort_values(time_col)
    diffs = sorted_df.groupby(entity_col)[time_col].diff()
    return diffs.reindex(df.index)
