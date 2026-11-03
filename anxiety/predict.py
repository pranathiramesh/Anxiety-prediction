"""Inference helpers shared by the CLI and the Streamlit app."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_regressor():
    if not config.REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"{config.REGRESSOR_PATH} not found. Run `python -m anxiety.train_regression` first."
        )
    return joblib.load(config.REGRESSOR_PATH)


def load_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text())


def severity_label(score: float) -> str:
    """Map a 1-10 score to its severity band (same edges as the classifier target)."""
    edges, labels = config.SEVERITY_BINS, config.SEVERITY_LABELS
    for upper, label in zip(edges[1:], labels):
        if score <= upper:
            return label
    return labels[-1]


def predict_anxiety(record: dict | pd.DataFrame) -> dict:
    """Takes raw feature values, returns raw + clipped score and severity band."""
    frame = pd.DataFrame([record]) if isinstance(record, dict) else record
    raw = float(load_regressor().predict(frame[config.RAW_FEATURES])[0])
    score = float(np.clip(raw, 1, 10))
    return {"raw_prediction": raw, "score": score, "severity": severity_label(score)}


if __name__ == "__main__":
    example = {
        "Age": 29, "Gender": "Female", "Occupation": "Artist", "Sleep Hours": 6.0,
        "Physical Activity (hrs/week)": 2.7, "Caffeine Intake (mg/day)": 181,
        "Alcohol Consumption (drinks/week)": 10, "Smoking": "Yes", "Family History of Anxiety": "No",
        "Stress Level (1-10)": 10, "Heart Rate (bpm)": 114, "Breathing Rate (breaths/min)": 14,
        "Sweating Level (1-5)": 4, "Dizziness": "No", "Medication": "Yes",
        "Therapy Sessions (per month)": 3, "Recent Major Life Event": "Yes", "Diet Quality (1-10)": 7,
    }
    print(predict_anxiety(example))
