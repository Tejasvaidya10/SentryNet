import pandas as pd
import pytest

from sentrynet.features.merchant_risk import MerchantRiskEncoder


def test_transform_before_fit_raises():
    encoder = MerchantRiskEncoder(category_col="ProductCD")
    with pytest.raises(RuntimeError):
        encoder.transform(pd.DataFrame({"ProductCD": ["W"]}))


def test_fit_transform_smooths_toward_global_rate():
    # Category "W": 1 fraud out of 2 (raw rate 0.5). Global rate: 2 fraud out of 6 (~0.333).
    # With smoothing=10, "W"'s encoded rate should sit between 0.5 and the global rate,
    # much closer to the global rate since count (2) << smoothing (10).
    train = pd.DataFrame(
        {
            "ProductCD": ["W", "W", "C", "C", "C", "C"],
            "isFraud": [1, 0, 1, 0, 0, 0],
        }
    )
    encoder = MerchantRiskEncoder(category_col="ProductCD", smoothing=10.0).fit(train)
    global_rate = 2 / 6
    expected_w = (0.5 * 2 + global_rate * 10.0) / (2 + 10.0)
    result = encoder.transform(pd.DataFrame({"ProductCD": ["W"]}))
    assert result.iloc[0] == pytest.approx(expected_w)


def test_transform_unseen_category_falls_back_to_global_rate():
    train = pd.DataFrame({"ProductCD": ["W", "W"], "isFraud": [1, 0]})
    encoder = MerchantRiskEncoder(category_col="ProductCD").fit(train)
    result = encoder.transform(pd.DataFrame({"ProductCD": ["UNSEEN"]}))
    assert result.iloc[0] == pytest.approx(0.5)
