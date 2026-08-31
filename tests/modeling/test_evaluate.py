import pytest

from sentrynet.modeling.evaluate import (
    pr_auc,
    precision_recall_at_threshold,
    select_threshold_by_cost,
)


def test_pr_auc_perfect_separation_is_one():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_precision_recall_at_threshold_hand_computed():
    y_true = [0, 1, 1, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    # threshold 0.5 -> predictions [1, 1, 0, 0]
    # tp=1 (idx1), fp=1 (idx0), fn=1 (idx2)
    precision, recall = precision_recall_at_threshold(y_true, y_score, threshold=0.5)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_select_threshold_by_cost_picks_zero_cost_threshold():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    best_threshold, best_cost = select_threshold_by_cost(
        y_true, y_score, cost_fp=1.0, cost_fn=1.0, thresholds=[0.3, 0.5, 0.7]
    )
    assert best_threshold == pytest.approx(0.5)
    assert best_cost == pytest.approx(0.0)
