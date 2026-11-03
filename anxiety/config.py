"""Single source of truth for paths, column names and modelling constants."""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "enhanced_anxiety_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REGRESSOR_PATH = MODELS_DIR / "anxiety_regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "anxiety_classifier.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
N_BOOTSTRAP = 5000

TARGET = "Anxiety Level (1-10)"

# Columns the user supplies (everything except the target).
RAW_FEATURES = [
    "Age",
    "Gender",
    "Occupation",
    "Sleep Hours",
    "Physical Activity (hrs/week)",
    "Caffeine Intake (mg/day)",
    "Alcohol Consumption (drinks/week)",
    "Smoking",
    "Family History of Anxiety",
    "Stress Level (1-10)",
    "Heart Rate (bpm)",
    "Breathing Rate (breaths/min)",
    "Sweating Level (1-5)",
    "Dizziness",
    "Medication",
    "Therapy Sessions (per month)",
    "Recent Major Life Event",
    "Diet Quality (1-10)",
]

# New column -> (source column, bin edges, labels)
CATEGORY_RULES = {
    "Sleep Category": (
        "Sleep Hours",
        [-np.inf, 5, 7, 9, np.inf],
        ["Very Low", "Low", "Optimal", "High"],
    ),
    "Activity Category": (
        "Physical Activity (hrs/week)",
        [-np.inf, 1, 3, 6, np.inf],
        ["Low", "Moderate", "High", "Very High"],
    ),
    "Caffeine Category": (
        "Caffeine Intake (mg/day)",
        [-np.inf, 100, 200, 400, np.inf],
        ["Low", "Moderate", "High", "Very High"],
    ),
    "Alcohol Category": (
        "Alcohol Consumption (drinks/week)",
        [-np.inf, 0, 3, 7, np.inf],
        ["None", "Low", "Moderate", "High"],
    ),
    "Stress Category": (
        "Stress Level (1-10)",
        [0, 3, 6, 10],
        ["Low", "Moderate", "High"],
    ),
    "Diet Category": (
        "Diet Quality (1-10)",
        [0, 3, 6, 10],
        ["Poor", "Moderate", "Good"],
    ),
}

# ---------------------------------------------------------------- severity
# Used both to build the classification target and to label regression scores.
SEVERITY_BINS = [0, 3, 6, 8, 10]
SEVERITY_LABELS = ["Low", "Moderate", "High", "Very High"]

# Tokens treated as missing during cleaning.
NULL_TOKENS = ["", " ", "NA", "N/A", "null", "None"]

# Columns used in EDA only.
PAIRPLOT_FEATURES = [
    "Age",
    "Sleep Hours",
    "Physical Activity (hrs/week)",
    "Stress Level (1-10)",
    "Heart Rate (bpm)",
    TARGET,
]
GROUP_FEATURES = [
    "Gender",
    "Occupation",
    "Smoking",
    "Family History of Anxiety",
    "Dizziness",
    "Medication",
    "Recent Major Life Event",
]
