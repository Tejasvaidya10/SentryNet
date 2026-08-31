import re

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_device_info(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    text = str(value).lower().strip()
    text = _NON_ALNUM.sub("_", text).strip("_")
    return text or "unknown"


def build_device_fingerprints(
    df: pd.DataFrame, device_info_col: str = "DeviceInfo", browser_col: str = "id_31"
) -> pd.Series:
    device = df[device_info_col].map(normalize_device_info)
    browser = df[browser_col].map(normalize_device_info)
    return "device:" + device + "|browser:" + browser


def build_card_entity_ids(
    df: pd.DataFrame,
    card_cols: tuple[str, ...] = ("card1", "card2", "card3", "card5"),
    addr_cols: tuple[str, ...] = ("addr1", "addr2"),
    d1_col: str = "D1",
    time_col: str = "TransactionDT",
    seconds_per_day: int = 86400,
) -> pd.Series:
    """Reconstruct a probabilistic 'same physical card' entity id.

    This is a heuristic, not a verified identity: it follows the public
    IEEE-CIS technique of combining card/address columns with an inferred
    card-origin day (derived from D1, days-since-card-first-seen, and the
    transaction time) to link transactions likely belonging to the same
    card. Two distinct cards that share card1/2/3/5+addr1/2 and were first
    used on the same day will collide into one entity id.
    """
    origin_day = (df[time_col] / seconds_per_day) - df[d1_col]
    origin_bucket = origin_day.round().astype("Int64").astype(str)
    key_cols = list(card_cols) + list(addr_cols)
    base = df[key_cols].astype(str).agg("_".join, axis=1)
    return "card:" + base + "|origin:" + origin_bucket
