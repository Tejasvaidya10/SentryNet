import numpy as np
import pandas as pd


def perturb_amount(
    df: pd.DataFrame, amount_col: str = "TransactionAmt", multiplier: float = 2.0
) -> pd.DataFrame:
    result = df.copy()
    result[amount_col] = result[amount_col] * multiplier
    return result


def perturb_category_mix(
    df: pd.DataFrame,
    category_col: str,
    target_category,
    target_share: float,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng if rng is not None else np.random.default_rng(42)
    result = df.copy()
    n = len(result)
    current_count = int((result[category_col] == target_category).sum())
    target_count = int(target_share * n)
    if target_count <= current_count:
        return result

    candidates = result.index[result[category_col] != target_category]
    n_to_flip = min(target_count - current_count, len(candidates))
    flip_idx = rng.choice(candidates, size=n_to_flip, replace=False)
    result.loc[flip_idx, category_col] = target_category
    return result


def degrade_feature_signal(
    df: pd.DataFrame, feature_col: str, noise_std: float, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    rng = rng if rng is not None else np.random.default_rng(42)
    result = df.copy()
    noise = rng.normal(0, noise_std, size=len(result))
    result[feature_col] = result[feature_col] + noise
    return result
