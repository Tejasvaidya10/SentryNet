import numpy as np
import pandas as pd

from sentrynet.drift.synthetic import (
    perturb_amount,
    perturb_category_mix,
    degrade_feature_signal,
)


def test_perturb_amount_multiplies_and_does_not_mutate_input():
    df = pd.DataFrame({"TransactionAmt": [10.0, 20.0]})
    result = perturb_amount(df, multiplier=2.0)
    assert result["TransactionAmt"].tolist() == [20.0, 40.0]
    assert df["TransactionAmt"].tolist() == [10.0, 20.0]


def test_perturb_category_mix_reaches_target_share():
    df = pd.DataFrame({"category": ["A"] * 90 + ["B"] * 10})
    result = perturb_category_mix(
        df, category_col="category", target_category="B", target_share=0.5, rng=np.random.default_rng(0)
    )
    assert (result["category"] == "B").sum() == 50


def test_perturb_category_mix_noop_if_already_above_target():
    df = pd.DataFrame({"category": ["B"] * 60 + ["A"] * 40})
    result = perturb_category_mix(
        df, category_col="category", target_category="B", target_share=0.5, rng=np.random.default_rng(0)
    )
    assert (result["category"] == "B").sum() == 60


def test_degrade_feature_signal_changes_values_reproducibly():
    df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    result_a = degrade_feature_signal(df, feature_col="feature", noise_std=1.0, rng=np.random.default_rng(0))
    result_b = degrade_feature_signal(df, feature_col="feature", noise_std=1.0, rng=np.random.default_rng(0))
    assert not result_a["feature"].equals(df["feature"])
    assert result_a["feature"].tolist() == result_b["feature"].tolist()
