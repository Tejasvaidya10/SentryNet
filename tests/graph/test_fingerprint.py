import numpy as np
import pandas as pd

from sentrynet.graph.fingerprint import (
    normalize_device_info,
    build_device_fingerprints,
    build_card_entity_ids,
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
