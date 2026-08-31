import pandas as pd


def transaction_entity_features(
    df: pd.DataFrame, entity_cols: tuple[str, ...], entity_features: pd.DataFrame
) -> pd.DataFrame:
    stat_cols = [c for c in entity_features.columns if c != "entity_key"]
    lookup = {
        row.entity_key: {stat: getattr(row, stat) for stat in stat_cols}
        for row in entity_features.itertuples(index=False)
    }

    result = pd.DataFrame(index=df.index)
    for col in entity_cols:
        keys = list(zip(["entity"] * len(df), [col] * len(df), df[col]))
        for stat in stat_cols:
            result[f"{col}_{stat}"] = [lookup.get(k, {}).get(stat) for k in keys]
    return result
