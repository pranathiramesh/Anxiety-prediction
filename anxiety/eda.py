"""Exploratory data analysis. Saves tables and figures instead of showing them.

Run:  python -m anxiety.eda
Outputs: reports/eda/*.csv  and  reports/figures/eda/*.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from . import config, data, plotting

TARGET = config.TARGET
TABLES = config.REPORTS_DIR / "eda"


def _table(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv")


def overview(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Duplicates:", df.duplicated().sum())
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_%": df.isnull().mean() * 100,
        "unique": df.nunique(),
    })
    print(summary.to_string())
    _table(summary, "overview")
    _table(df.describe(include="all").T, "summary_statistics")


def distributions(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    df[numeric].hist(figsize=(18, 15), bins=30, edgecolor="black")
    plt.suptitle("Numerical feature distributions", fontsize=18)
    plotting.save("numeric_distributions.png", "eda")

    for col in categorical:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}"); plt.xticks(rotation=45, ha="right")
        plotting.save(f"count_{plotting.slug(col)}.png", "eda/categorical")


def target_analysis(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[TARGET], bins=10, kde=True)
    plt.title("Distribution of anxiety levels")
    plotting.save("target_distribution.png", "eda")

    print("Target stats:\n", df[TARGET].describe().to_string())
    print(f"Skewness: {df[TARGET].skew():.3f}  Kurtosis: {df[TARGET].kurtosis():.3f}")


def relationships(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    for col in numeric:
        if col == TARGET:
            continue
        plt.figure(figsize=(8, 5))
        sns.regplot(data=df, x=col, y=TARGET, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
        plt.title(f"{col} vs anxiety")
        plotting.save(f"scatter_{plotting.slug(col)}.png", "eda/numeric_vs_target")

    for col in categorical:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=col, y=TARGET)
        plt.title(f"{col} vs anxiety"); plt.xticks(rotation=45, ha="right")
        plotting.save(f"box_{plotting.slug(col)}.png", "eda/categorical_vs_target")

    for col in config.GROUP_FEATURES:
        table = df.groupby(col, observed=True)[TARGET].agg(["count", "mean", "median", "std"])
        _table(table.sort_values("mean", ascending=False), f"group_{plotting.slug(col)}")


def correlations(df: pd.DataFrame, numeric: list[str]) -> None:
    for method, cmap, kwargs in [("pearson", "coolwarm", {}), ("spearman", "RdBu_r", {"center": 0})]:
        corr = df[numeric].corr(method=method)
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, **kwargs)
        plt.title(f"{method.title()} correlation")
        plotting.save(f"{method}_correlation.png", "eda")
        if method == "pearson":
            with_target = corr[TARGET].drop(TARGET).sort_values(ascending=False).to_frame("correlation")
            print("\nCorrelation with target:\n", with_target.to_string())
            _table(with_target, "correlation_with_target")


def outliers_and_distribution(df: pd.DataFrame, numeric: list[str]) -> None:
    rows, dist_rows, normal_rows = [], [], []
    for col in numeric:
        values = df[col].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((values < low) | (values > high)).sum())
        rows.append({"Feature": col, "Lower": low, "Upper": high,
                     "Outliers": n_out, "Outlier_%": n_out / len(values) * 100})
        dist_rows.append({"Feature": col, "Mean": values.mean(), "Median": values.median(),
                          "Mode": values.mode().iloc[0], "Variance": values.var(),
                          "Std": values.std(), "Skew": stats.skew(values), "Kurtosis": stats.kurtosis(values)})
        sample = values.sample(min(len(values), 5000), random_state=config.RANDOM_STATE)
        stat, p = stats.normaltest(sample)
        normal_rows.append({"Feature": col, "Statistic": stat, "P_Value": p, "Normal (p>0.05)": p > 0.05})

    for name, rows_ in [("iqr_outliers", rows), ("distribution_stats", dist_rows), ("normality_tests", normal_rows)]:
        frame = pd.DataFrame(rows_).set_index("Feature")
        _table(frame, name)
        print(f"\n{name}:\n", frame.round(3).to_string())

    fig, axes = plt.subplots(len(numeric), 1, figsize=(10, 2.2 * len(numeric)))
    for ax, col in zip(np.atleast_1d(axes), numeric):
        sns.boxplot(x=df[col], ax=ax)
    plotting.save("outlier_boxplots.png", "eda")


def vif(df: pd.DataFrame, numeric: list[str]) -> None:
    """Variance inflation factor via R^2 of each feature on the others (no statsmodels needed)."""
    features = [c for c in numeric if c != TARGET]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[features]), columns=features)
    rows = []
    for col in features:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        rows.append({"Feature": col, "VIF": 1 / (1 - r2) if r2 < 1 else np.inf})
    table = pd.DataFrame(rows).set_index("Feature").sort_values("VIF", ascending=False)
    print("\nVIF:\n", table.round(3).to_string())
    _table(table, "vif")


def pairplot(df: pd.DataFrame) -> None:
    sample = df[config.PAIRPLOT_FEATURES].sample(min(len(df), 2000), random_state=config.RANDOM_STATE)
    sns.pairplot(sample, corner=True, diag_kind="kde", plot_kws={"alpha": 0.4})
    plotting.save("pairplot.png", "eda")


def main() -> None:
    df = data.clean(data.load_raw())
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    overview(df)
    distributions(df, numeric, categorical)
    target_analysis(df)
    relationships(df, numeric, categorical)
    correlations(df, numeric)
    outliers_and_distribution(df, numeric)
    vif(df, numeric)
    pairplot(df)
    print(f"\nTables  -> {TABLES}\nFigures -> {config.FIGURES_DIR / 'eda'}")


if __name__ == "__main__":
    main()
