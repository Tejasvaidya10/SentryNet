import pandas as pd
import xgboost as xgb


def compute_scale_pos_weight(y: pd.Series) -> float:
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0:
        raise ValueError("y contains no positive examples")
    return negative / positive


def train_model(
    X: pd.DataFrame, y: pd.Series, params: dict | None = None
) -> xgb.XGBClassifier:
    model_params = {
        "scale_pos_weight": compute_scale_pos_weight(y),
        "eval_metric": "aucpr",
        "random_state": 42,
    }
    if params:
        model_params.update(params)
    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y)
    return model
