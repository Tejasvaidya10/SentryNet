import numpy as np
import pandas as pd
import pytest

from sentrynet.modeling.train import compute_scale_pos_weight, train_model


def test_compute_scale_pos_weight():
    y = pd.Series([1, 0, 0, 0])
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)


def test_compute_scale_pos_weight_raises_with_no_positives():
    y = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(y)


def test_train_model_fits_and_predicts():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = pd.Series([1] * 5 + [0] * 35)
    model = train_model(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (40, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert model.get_params()["scale_pos_weight"] == pytest.approx(7.0)
