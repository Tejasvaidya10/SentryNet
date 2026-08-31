import pandas as pd


class MerchantRiskEncoder:
    """Smoothed (Bayesian) mean-encoding of a category's historical fraud rate.

    Must be fit on the training split only and applied via transform() to
    both train and test, to avoid leaking test-set fraud labels into the
    encoding.
    """

    def __init__(self, category_col: str, label_col: str = "isFraud", smoothing: float = 10.0):
        self.category_col = category_col
        self.label_col = label_col
        self.smoothing = smoothing
        self.global_rate_: float | None = None
        self.rates_: dict | None = None

    def fit(self, df: pd.DataFrame) -> "MerchantRiskEncoder":
        self.global_rate_ = df[self.label_col].mean()
        grouped = df.groupby(self.category_col)[self.label_col].agg(["mean", "count"])
        smoothed = (
            grouped["mean"] * grouped["count"] + self.global_rate_ * self.smoothing
        ) / (grouped["count"] + self.smoothing)
        self.rates_ = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.rates_ is None:
            raise RuntimeError("call fit() before transform()")
        return df[self.category_col].map(self.rates_).fillna(self.global_rate_)
