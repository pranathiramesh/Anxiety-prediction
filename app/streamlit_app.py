"""Streamlit front-end.  Run from the project root:

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `anxiety` importable

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from anxiety import config  # noqa: E402
from anxiety.predict import load_metadata, load_regressor, predict_anxiety  # noqa: E402

SEVERITY_COLORS = {"Low": "#2E7D32", "Moderate": "#FBC02D", "High": "#F57C00", "Very High": "#C62828"}

st.set_page_config(page_title="Anxiety Level Analytics", page_icon="🧠", layout="wide")
st.title("🧠 Anxiety Level Analytics & Classification")
st.write(
    "Enter personal, lifestyle and health indicators to estimate a numerical anxiety "
    "level and its severity category."
)

try:
    load_regressor()
    options = load_metadata()["feature_options"]
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


def number(col, label=None, step=1.0, container=st):
    spec = options[col]
    is_float = step < 1 or isinstance(spec["median"], float) and not float(spec["median"]).is_integer()
    cast = float if is_float else int
    return container.number_input(
        label or col,
        min_value=cast(18 if col == "Age" else spec["min"] * 0.5),
        max_value=cast(spec["max"] * 2) if col != "Age" else 100,
        value=cast(spec["median"]),
        step=cast(step),
    )


def choice(col, container=st):
    return container.selectbox(col, options[col]["values"])


def rating(col, container=st):
    spec = options[col]
    return container.slider(col, int(spec["min"]), int(spec["max"]), int(round(spec["median"])))


with st.form("anxiety_form"):
    st.subheader("A. Demographics")
    c1, c2, c3 = st.columns(3)
    age = number("Age", container=c1)
    gender = choice("Gender", c2)
    occupation = choice("Occupation", c3)

    st.divider()
    st.subheader("B. Lifestyle")
    c1, c2, c3 = st.columns(3)
    sleep = number("Sleep Hours", step=0.5, container=c1)
    activity = number("Physical Activity (hrs/week)", step=0.5, container=c1)
    caffeine = number("Caffeine Intake (mg/day)", step=10, container=c2)
    alcohol = number("Alcohol Consumption (drinks/week)", container=c2)
    smoking = choice("Smoking", c3)
    diet = rating("Diet Quality (1-10)", c3)

    st.divider()
    st.subheader("C. Health and history")
    c1, c2, c3 = st.columns(3)
    stress = rating("Stress Level (1-10)", c1)
    heart_rate = number("Heart Rate (bpm)", container=c1)
    breathing = number("Breathing Rate (breaths/min)", container=c1)
    sweating = rating("Sweating Level (1-5)", c2)
    dizziness = choice("Dizziness", c2)
    medication = choice("Medication", c2)
    family = choice("Family History of Anxiety", c3)
    therapy = number("Therapy Sessions (per month)", container=c3)
    life_event = choice("Recent Major Life Event", c3)

    submitted = st.form_submit_button("Run analysis", use_container_width=True)

if submitted:
    record = {
        "Age": age, "Gender": gender, "Occupation": occupation, "Sleep Hours": sleep,
        "Physical Activity (hrs/week)": activity, "Caffeine Intake (mg/day)": caffeine,
        "Alcohol Consumption (drinks/week)": alcohol, "Smoking": smoking,
        "Family History of Anxiety": family, "Stress Level (1-10)": stress,
        "Heart Rate (bpm)": heart_rate, "Breathing Rate (breaths/min)": breathing,
        "Sweating Level (1-5)": sweating, "Dizziness": dizziness, "Medication": medication,
        "Therapy Sessions (per month)": therapy, "Recent Major Life Event": life_event,
        "Diet Quality (1-10)": diet,
    }
    result = predict_anxiety(record)
    score, severity = result["score"], result["severity"]

    st.divider()
    st.header("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated severity index", f"{score:.2f} / 10")
    c2.markdown(
        f"<div style='padding:10px;border-radius:5px;background:{SEVERITY_COLORS[severity]};"
        f"text-align:center;color:white;font-weight:bold;font-size:24px;'>{severity} anxiety level</div>",
        unsafe_allow_html=True,
    )
    c3.metric("Raw model output", f"{result['raw_prediction']:.4f}")
    st.progress(min(100, int(round(score * 10))))

    fig, ax = plt.subplots(figsize=(8, 2))
    spans = zip(config.SEVERITY_BINS[:-1], config.SEVERITY_BINS[1:], config.SEVERITY_LABELS)
    for lo, hi, label in spans:
        ax.axvspan(lo, hi, color=SEVERITY_COLORS[label], alpha=0.25, label=f"{label} ({lo}-{hi})")
    ax.axvline(score, color="darkblue", ls="--", lw=2)
    ax.scatter([score], [0.5], color="darkblue", s=150, zorder=5, label="Your score")
    ax.set_yticks([]); ax.set_xlim(0, 10); ax.set_xlabel("Anxiety scale")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=5, fontsize=8)
    st.pyplot(fig)

    with st.expander("Entered values"):
        st.dataframe(pd.DataFrame([record]), use_container_width=True)

    st.caption("This is a machine-learning estimate, not a clinical diagnosis.")
    st.info("If you or someone you know is struggling with persistent anxiety, please speak with a healthcare professional.")
