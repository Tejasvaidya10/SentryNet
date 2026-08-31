import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def precision_recall_at_threshold(y_true, y_score, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def select_threshold_by_cost(
    y_true, y_score, cost_fp: float, cost_fn: float, thresholds=None
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    best_threshold = None
    best_cost = float("inf")
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = fp * cost_fp + fn * cost_fn
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t
    return best_threshold, best_cost
