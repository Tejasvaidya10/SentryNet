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
    """Build a per-transaction device/browser fingerprint.

    Returns NaN when both the device and browser signal are unknown, rather
    than the literal string "device:unknown|browser:unknown" -- otherwise
    every transaction with no identity data at all (the majority of rows in
    IEEE-CIS, since only ~24% of transactions have any identity record)
    would collide into a single supernode entity. Downstream graph
    construction (see build_bipartite_graph) must skip NaN entity values so
    "no signal" is never treated as a shared identity.
    """
    device = df[device_info_col].map(normalize_device_info)
    browser = df[browser_col].map(normalize_device_info)
    fingerprint = "device:" + device + "|browser:" + browser
    has_signal = (device != "unknown") | (browser != "unknown")
    return fingerprint.where(has_signal)


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
    # Missing values are filled with an out-of-range sentinel *before* casting
    # to str: pandas' astype(str) does not stringify NaN/NA into "nan" (it
    # leaves them as missing), which breaks the row-wise join below if left
    # in place.
    origin_day = (df[time_col] / seconds_per_day) - df[d1_col]
    origin_bucket = origin_day.round().astype("Int64").fillna(-999999).astype(str)
    key_cols = list(card_cols) + list(addr_cols)
    base = df[key_cols].fillna(-1).astype(str).agg("_".join, axis=1)
    return "card:" + base + "|origin:" + origin_bucket


def cap_high_frequency_entities(entity_values: pd.Series, max_count: int = 1000) -> pd.Series:
    """Null out entity values shared by more than max_count transactions.

    An entity that generic is far more likely to be a common, unremarkable
    signature (e.g. a popular "Windows + Chrome" combination) than a
    genuine shared identity -- and including it as a graph node would make
    bipartite-graph projection (see extract_entity_features) computationally
    intractable, since projection cost scales with the square of each
    entity's degree.
    """
    counts = entity_values.value_counts()
    high_frequency = counts[counts > max_count].index
    return entity_values.where(~entity_values.isin(high_frequency))
