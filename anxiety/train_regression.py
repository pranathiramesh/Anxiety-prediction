"""Train, cross-validate, tune and save the anxiety-score regressor.

Run:  python -m anxiety.train_regression
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.svm import SVR

from . import config, data, plotting
from .pipeline import build_pipeline

RS = config.RANDOM_STATE


def candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RS
        ),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
    }


PARAM_GRIDS = {
    "Linear Regression": {},
    "Elastic Net": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.2, 0.5, 0.8],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3, 4],
    },
    "SVR": {
        "model__C": [1, 10, 50],
        "model__epsilon": [0.05, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    },
}


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def feature_options(df: pd.DataFrame) -> dict:
    """Allowed values / ranges per input, so the app never hard-codes them."""
    options = {}
    for col in config.RAW_FEATURES:
        if pd.api.types.is_numeric_dtype(df[col]):
            options[col] = {"type": "numeric", "min": float(df[col].min()),
                            "max": float(df[col].max()), "median": float(df[col].median())}
        else:
            options[col] = {"type": "categorical", "values": sorted(df[col].dropna().unique().tolist())}
    return options


def main() -> None:
    df = data.clean(data.load_raw())
    X, y = data.split_xy(df)
    X_train, X_test, y_train, y_test = data.train_test(X, y)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    models = candidate_models()
    cv = KFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=RS)
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

    # 1) hold-out test scores + 2) cross-validation on the training split
    holdout_rows, cv_rows = [], []
    for name, model in models.items():
        print(f"[{name}] fitting + cross-validating ...")
        pipe = build_pipeline(model).fit(X_train, y_train)
        holdout_rows.append({"Model": name, **regression_metrics(y_test, pipe.predict(X_test))})

        scores = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows.append({
            "Model": name,
            "Mean MAE": -scores["test_MAE"].mean(),
            "Mean RMSE": -scores["test_RMSE"].mean(),
            "Mean R2": scores["test_R2"].mean(),
            "RMSE Std": scores["test_RMSE"].std(),
        })

    holdout = pd.DataFrame(holdout_rows).sort_values("RMSE").reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values("Mean RMSE").reset_index(drop=True)
    print("\nHold-out comparison:\n", holdout.round(4).to_string(index=False))
    print("\n5-fold CV comparison:\n", cv_df.round(4).to_string(index=False))

    # 3) tune the model with the best CV RMSE
    best_name = cv_df.loc[0, "Model"]
    print(f"\nBest model by CV RMSE: {best_name}")
    pipe = build_pipeline(models[best_name])
    grid = PARAM_GRIDS[best_name]
    if grid:
        search = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        final_model, best_params = search.best_estimator_, search.best_params_
        print("Best params:", best_params, " CV RMSE:", round(-search.best_score_, 4))
    else:
        final_model, best_params = pipe.fit(X_train, y_train), {}

    # 4) final evaluation on the untouched test split
    preds = final_model.predict(X_test)
    final = regression_metrics(y_test, preds)
    print(f"\nFINAL MODEL: {best_name}")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    # 5) save artefacts
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, config.REGRESSOR_PATH)
    holdout.to_csv(config.REPORTS_DIR / "regression_holdout_metrics.csv", index=False)
    cv_df.to_csv(config.REPORTS_DIR / "regression_cv_metrics.csv", index=False)
    config.METADATA_PATH.write_text(json.dumps({
        "best_model": best_name,
        "best_params": best_params,
        "final_test_metrics": final,
        "raw_features": config.RAW_FEATURES,
        "severity_bins": config.SEVERITY_BINS,
        "severity_labels": config.SEVERITY_LABELS,
        "feature_options": feature_options(df),
    }, indent=2, default=str))

    # 6) figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, ["MAE", "MSE", "RMSE", "R2"]):
        sns.barplot(data=holdout, x="Model", y=metric, ax=ax)
        ax.set_title(f"Hold-out {metric}")
        ax.tick_params(axis="x", rotation=35)
    plotting.save("model_comparison.png", "regression")

    plt.figure(figsize=(10, 5))
    plt.bar(cv_df["Model"], cv_df["Mean RMSE"], yerr=cv_df["RMSE Std"], capsize=4)
    plt.ylabel("Mean RMSE"); plt.title("5-fold CV RMSE"); plt.xticks(rotation=30)
    plotting.save("cv_rmse.png", "regression")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Actual vs Predicted ({best_name})")
    plotting.save("actual_vs_predicted.png", "regression")

    residuals = y_test - preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, kde=True, bins=30, ax=axes[0]); axes[0].set_title("Residual distribution")
    axes[1].scatter(preds, residuals, alpha=0.5); axes[1].axhline(0, ls="--")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residuals vs predicted")
    plotting.save("residuals.png", "regression")

    print(f"\nSaved model -> {config.REGRESSOR_PATH}")


if __name__ == "__main__":
    main()
