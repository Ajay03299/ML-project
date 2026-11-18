"""Simple forecasting baselines with a scikit-learn-like fit/predict API.   Owner: Ajay.

The TA asks for "meaningful baselines". Any ML model must beat these to be worth using.
They need the full feature set (run them with all feature groups).
"""
import pandas as pd


class ColumnBaseline:
    """Predict by copying one feature column.

    flow_lag0       -> persistence (the next hour looks like right now)
    flow_yesterday  -> seasonal naive, same time yesterday
    flow_lastweek   -> seasonal naive, same time last week
    """

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.column].to_numpy()


class HistoricalAverage:
    """Mean flow per sensor x 15-minute slot x weekday/weekend, learned on training data."""

    KEYS = ["sensor_id", "slot", "is_weekend"]

    def fit(self, X, y):
        df = X[self.KEYS].astype("int32").copy()
        df["y"] = pd.Series(y).to_numpy()
        self.table_ = df.groupby(self.KEYS)["y"].mean()
        self.fallback_ = float(df["y"].mean())
        return self

    def predict(self, X):
        key = pd.MultiIndex.from_frame(X[self.KEYS].astype("int32"))
        return self.table_.reindex(key).fillna(self.fallback_).to_numpy()
