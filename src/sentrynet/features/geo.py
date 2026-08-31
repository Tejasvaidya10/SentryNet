import pandas as pd


def addr_change_flag(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    addr_cols: tuple[str, ...] = ("addr1", "addr2"),
) -> pd.Series:
    sorted_df = df.sort_values(time_col)
    changed = pd.Series(False, index=sorted_df.index)
    for col in addr_cols:
        prev = sorted_df.groupby(entity_col)[col].shift()
        changed = changed | (prev.notna() & sorted_df[col].notna() & (prev != sorted_df[col]))
    return changed.reindex(df.index)
