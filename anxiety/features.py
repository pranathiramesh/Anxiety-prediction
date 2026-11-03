"""Custom scikit-learn transformers.

Both transformers live *inside* the saved pipeline, so training and serving
run exactly the same code path (no copy-pasted logic in the app).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds binned categories and the Heart Rate per Age ratio."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for new_col, (source, bins, labels) in config.CATEGORY_RULES.items():
            X[new_col] = pd.cut(X[source], bins=bins, labels=labels).astype(object)

        X["Heart Rate per Age"] = X["Heart Rate (bpm)"] / X["Age"].replace(0, np.nan)
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [Q1 - k*IQR, Q3 + k*IQR].

    Bounds are learned on the training split only, so no information from the
    test set leaks into preprocessing.
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X: pd.DataFrame, y=None):
        numeric = X.select_dtypes(include=np.number)
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.k * iqr).to_dict()
        self.upper_ = (q3 + self.k * iqr).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, low in self.lower_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low, upper=self.upper_[col])
        return X
