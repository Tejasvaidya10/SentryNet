import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    ref_counts, _ = np.histogram(reference, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    comp_pct = comp_counts / comp_counts.sum()

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    comp_pct = np.clip(comp_pct, epsilon, None)

    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def ks_drift_test(reference: np.ndarray, comparison: np.ndarray) -> tuple[float, float]:
    statistic, p_value = ks_2samp(reference, comparison)
    return float(statistic), float(p_value)


def is_drifted(psi_value: float, threshold: float = 0.2) -> bool:
    return psi_value > threshold
