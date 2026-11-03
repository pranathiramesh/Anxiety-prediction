"""Builds the full preprocessing + model pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureEngineer, IQRClipper


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def build_pipeline(estimator) -> Pipeline:
    """raw columns -> feature engineering -> IQR clipping -> encode/scale -> model."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("clipper", IQRClipper()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
