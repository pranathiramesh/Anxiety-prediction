"""Classify anxiety severity (Low / Moderate / High / Very High).

Compares a Logistic Regression baseline against a Random Forest, then runs an
exact McNemar test and a paired bootstrap on macro-F1.

Run:  python -m anxiety.train_classification
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from . import config, data, plotting
from .pipeline import build_pipeline

RS = config.RANDOM_STATE
LABELS = config.SEVERITY_LABELS


def summarise(name: str, y_true, y_pred) -> dict:
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def mcnemar_exact(y_true, pred_a, pred_b) -> tuple[int, int, float]:
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    p = binomtest(min(only_a, only_b), n=only_a + only_b, p=0.5).pvalue
    return only_a, only_b, float(p)


def bootstrap_f1_difference(y_true, pred_base, pred_new, n_boot: int = config.N_BOOTSTRAP):
    y_true, pred_base, pred_new = map(np.asarray, (y_true, pred_base, pred_new))

    def macro_f1(t, p):
        return f1_score(t, p, labels=LABELS, average="macro", zero_division=0)

    observed = macro_f1(y_true, pred_new) - macro_f1(y_true, pred_base)
    rng = np.random.default_rng(RS)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = macro_f1(y_true[idx], pred_new[idx]) - macro_f1(y_true[idx], pred_base[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return observed, float(low), float(high)


def main() -> None:
    df = data.clean(data.load_raw())
    X, _ = data.split_xy(df)
    y = data.make_severity_labels(df[config.TARGET])

    print("Class distribution (%):")
    print((y.value_counts(normalize=True) * 100).round(2).to_string())

    X_train, X_test, y_train, y_test = data.train_test(X, y, stratify=y)

    baseline = build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RS))
    improved = build_pipeline(RandomForestClassifier(
        n_estimators=500, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RS, n_jobs=-1))

    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)
    pred_base, pred_new = baseline.predict(X_test), improved.predict(X_test)

    comparison = pd.DataFrame([
        summarise("Logistic Regression", y_test, pred_base),
        summarise("Random Forest", y_test, pred_new),
    ])
    print("\n", comparison.round(4).to_string(index=False))
    for name, pred in [("LOGISTIC REGRESSION", pred_base), ("RANDOM FOREST", pred_new)]:
        print(f"\n{name} - classification report")
        print(classification_report(y_test, pred, labels=LABELS, zero_division=0))

    only_base, only_new, p_value = mcnemar_exact(y_test, pred_base, pred_new)
    observed, low, high = bootstrap_f1_difference(y_test, pred_base, pred_new)
    print(f"McNemar exact: baseline-only correct={only_base}, RF-only correct={only_new}, p={p_value:.6f}")
    print(f"Macro-F1 difference (RF - LR): {observed:.4f}  95% CI ({low:.4f}, {high:.4f})")

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(improved, config.CLASSIFIER_PATH)
    comparison.to_csv(config.REPORTS_DIR / "classification_metrics.csv", index=False)
    pd.DataFrame([{"mcnemar_p": p_value, "f1_diff": observed, "ci_low": low, "ci_high": high}]).to_csv(
        config.REPORTS_DIR / "classification_significance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (name, pred) in zip(axes, [("Logistic Regression", pred_base), ("Random Forest", pred_new)]):
        sns.heatmap(confusion_matrix(y_test, pred, labels=LABELS), annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plotting.save("confusion_matrices.png", "classification")
    print(f"\nSaved classifier -> {config.CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
