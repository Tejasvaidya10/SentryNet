import numpy as np
import pandas as pd

from sentrynet.graph.fingerprint import (
    normalize_device_info,
    build_device_fingerprints,
    build_card_entity_ids,
    cap_high_frequency_entities,
)


def test_normalize_device_info_lowercases_and_strips_punctuation():
    assert normalize_device_info("SM-G935F Build/NRD90M") == "sm_g935f_build_nrd90m"


def test_normalize_device_info_handles_missing():
    assert normalize_device_info(None) == "unknown"
    assert normalize_device_info(np.nan) == "unknown"


def test_build_device_fingerprints_groups_identical_normalized_values():
    df = pd.DataFrame(
        {
            "DeviceInfo": ["iOS Device", "ios device", "Windows"],
            "id_31": ["safari", "Safari", "chrome"],
        }
    )
    fingerprints = build_device_fingerprints(df)
    assert fingerprints.iloc[0] == fingerprints.iloc[1]
    assert fingerprints.iloc[0] != fingerprints.iloc[2]


def test_build_device_fingerprints_returns_nan_when_no_signal_at_all():
    # Most IEEE-CIS transactions have no identity record at all (DeviceInfo
    # and id_31 both missing). These must come back as NaN, not the literal
    # string "device:unknown|browser:unknown", or every such transaction
    # would collide into one supernode entity in the graph.
    df = pd.DataFrame(
        {
            "DeviceInfo": [np.nan, "Windows", np.nan],
            "id_31": [np.nan, np.nan, np.nan],
        }
    )
    fingerprints = build_device_fingerprints(df)
    assert pd.isna(fingerprints.iloc[0])
    assert fingerprints.iloc[1] == "device:windows|browser:unknown"
    assert pd.isna(fingerprints.iloc[2])


def test_build_card_entity_ids_same_card_same_origin_day():
    # Same card1-5/addr, and D1/TransactionDT imply the same origin day -> same entity id.
    df = pd.DataFrame(
        {
            "card1": [100, 100],
            "card2": [200, 200],
            "card3": [150, 150],
            "card5": [226, 226],
            "addr1": [300, 300],
            "addr2": [87, 87],
            "D1": [0, 1],
            "TransactionDT": [0, 86400],  # day 0 and day 1, D1 0 and 1 -> same origin day 0
        }
    )
    ids = build_card_entity_ids(df)
    assert ids.iloc[0] == ids.iloc[1]


def test_build_card_entity_ids_different_origin_day_differ():
    df = pd.DataFrame(
        {
            "card1": [100, 100],
            "card2": [200, 200],
            "card3": [150, 150],
            "card5": [226, 226],
            "addr1": [300, 300],
            "addr2": [87, 87],
            "D1": [0, 0],
            "TransactionDT": [0, 864000],  # day 10, D1 0 -> origin day 10, differs from origin day 0
        }
    )
    ids = build_card_entity_ids(df)
    assert ids.iloc[0] != ids.iloc[1]


def test_build_card_entity_ids_handles_missing_values_without_raising():
    # card2/card3/card5/addr1/addr2/D1 are all frequently missing in the
    # real IEEE-CIS data; this must not raise and must still produce a
    # distinct id per row rather than colliding all-NaN rows together.
    df = pd.DataFrame(
        {
            "card1": [100, 200],
            "card2": [np.nan, np.nan],
            "card3": [150, np.nan],
            "card5": [np.nan, 226],
            "addr1": [np.nan, np.nan],
            "addr2": [np.nan, np.nan],
            "D1": [np.nan, 5],
            "TransactionDT": [0, 432000],
        }
    )
    ids = build_card_entity_ids(df)
    assert ids.notna().all()
    assert (ids.map(type) == str).all()
    assert ids.iloc[0] != ids.iloc[1]


def test_cap_high_frequency_entities_nulls_only_values_above_threshold():
    values = pd.Series(["common"] * 5 + ["rare"] * 2 + ["also_rare"] * 2)
    result = cap_high_frequency_entities(values, max_count=3)
    assert result[:5].isna().all()
    assert (result[5:7] == "rare").all()
    assert (result[7:9] == "also_rare").all()
