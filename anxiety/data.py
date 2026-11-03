"""Loading, cleaning and splitting the dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, normalise null tokens, drop duplicates / constant columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().astype(object)

    df = df.replace(config.NULL_TOKENS, np.nan)
    df = df.drop_duplicates()

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=constant)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.RAW_FEATURES], df[config.TARGET]


def make_severity_labels(scores: pd.Series) -> pd.Series:
    """Bin 1-10 anxiety scores into Low / Moderate / High / Very High."""
    return pd.cut(
        scores,
        bins=config.SEVERITY_BINS,
        labels=config.SEVERITY_LABELS,
        include_lowest=True,
    )


def train_test(X, y, stratify=None):
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
