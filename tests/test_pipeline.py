import numpy as np
import pandas as pd

from anxiety import config, data
from anxiety.features import FeatureEngineer, IQRClipper
from anxiety.predict import severity_label


def sample_frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Age": rng.integers(18, 65, n),
        "Sleep Hours": rng.uniform(3, 10, n),
        "Physical Activity (hrs/week)": rng.uniform(0, 8, n),
        "Caffeine Intake (mg/day)": rng.integers(0, 500, n),
        "Alcohol Consumption (drinks/week)": rng.integers(0, 15, n),
        "Stress Level (1-10)": rng.integers(1, 11, n),
        "Diet Quality (1-10)": rng.integers(1, 11, n),
        "Heart Rate (bpm)": rng.integers(60, 120, n),
    })


def test_feature_engineer_adds_expected_columns():
    out = FeatureEngineer().transform(sample_frame())
    for col in list(config.CATEGORY_RULES) + ["Heart Rate per Age"]:
        assert col in out.columns
    assert not out["Sleep Category"].isna().any()


def test_sleep_bins_match_notebook():
    df = sample_frame(4)
    df["Sleep Hours"] = [4.0, 6.0, 8.0, 10.0]
    out = FeatureEngineer().transform(df)
    assert out["Sleep Category"].tolist() == ["Very Low", "Low", "Optimal", "High"]


def test_zero_age_gives_nan_not_inf():
    df = sample_frame(3)
    df["Age"] = [0, 30, 40]
    out = FeatureEngineer().transform(df)
    assert np.isnan(out.loc[0, "Heart Rate per Age"])


def test_iqr_clipper_learns_bounds_on_fit_only():
    train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    clipper = IQRClipper().fit(train)
    out = clipper.transform(pd.DataFrame({"x": [-1000, 5, 1000]}))
    assert out["x"].min() > -1000 and out["x"].max() < 1000
    assert out.loc[1, "x"] == 5


def test_severity_label_edges():
    assert severity_label(1) == "Low"
    assert severity_label(3) == "Low"
    assert severity_label(3.01) == "Moderate"
    assert severity_label(6) == "Moderate"
    assert severity_label(8) == "High"
    assert severity_label(8.5) == "Very High"


def test_severity_labels_agree_with_binning():
    scores = pd.Series(np.arange(1, 10.01, 0.25))
    binned = data.make_severity_labels(scores).astype(str).tolist()
    assert binned == [severity_label(s) for s in scores]


def test_clean_strips_and_drops_constants():
    df = pd.DataFrame({" a ": [" x", "y ", "y "], "b": [1, 1, 1], "c": [1, 2, 2]})
    out = data.clean(df)
    assert "a" in out.columns and "b" not in out.columns
    assert set(out["a"]) == {"x", "y"}
