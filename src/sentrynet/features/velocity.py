import numpy as np
import pandas as pd


def transaction_velocity(
    df: pd.DataFrame, entity_col: str, time_col: str, window_seconds: float
) -> pd.Series:
    """Count prior transactions by the same entity within window_seconds before each row."""
    result = pd.Series(0, index=df.index, dtype="int64")
    for _, group in df.groupby(entity_col):
        times = group[time_col].to_numpy()
        order = np.argsort(times, kind="mergesort")
        sorted_times = times[order]
        sorted_idx = group.index.to_numpy()[order]
        counts = np.zeros(len(sorted_times), dtype="int64")
        start = 0
        for i, t in enumerate(sorted_times):
            while sorted_times[start] < t - window_seconds:
                start += 1
            counts[i] = i - start
        result.loc[sorted_idx] = counts
    return result.reindex(df.index)
