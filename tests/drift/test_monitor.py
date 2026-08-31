import numpy as np

from sentrynet.drift.monitor import population_stability_index, ks_drift_test, is_drifted


def test_psi_is_near_zero_for_identical_distributions():
    reference = np.tile(np.arange(10), 100).astype(float)
    psi = population_stability_index(reference, reference, bins=10)
    assert abs(psi) < 1e-6


def test_psi_is_large_for_a_distribution_that_collapses_to_one_bin():
    reference = np.tile(np.arange(10), 100).astype(float)
    comparison = np.full(1000, 9.0)
    psi = population_stability_index(reference, comparison, bins=10)
    assert psi > 0.2


def test_is_drifted_thresholding():
    assert is_drifted(0.25, threshold=0.2) is True
    assert is_drifted(0.1, threshold=0.2) is False


def test_ks_drift_test_detects_shifted_distribution():
    reference = np.tile(np.arange(10), 100).astype(float)
    comparison = np.full(1000, 9.0)
    statistic, p_value = ks_drift_test(reference, comparison)
    assert statistic > 0.5
    assert p_value < 0.01
