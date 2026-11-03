# Anxiety Level Prediction

Predicts a 1-10 anxiety score (and a Low / Moderate / High / Very High severity band)
from lifestyle, physiological and personal-history features.
Refactored from the original Colab notebook (`notebooks/original_colab_notebook.ipynb`).

## Layout

```
anxiety-prediction/
├── anxiety/                    # importable package
│   ├── config.py               # paths, column names, bins, constants (edit here first)
│   ├── data.py                 # load / clean / split / severity labels
│   ├── features.py             # FeatureEngineer + IQRClipper (sklearn transformers)
│   ├── pipeline.py             # full pipeline: engineer -> clip -> encode/scale -> model
│   ├── eda.py                  # tables -> reports/eda, figures -> reports/figures/eda
│   ├── train_regression.py     # 5 models, CV, grid search, saves best
│   ├── train_classification.py # LogReg vs RandomForest + McNemar + bootstrap F1
│   ├── predict.py              # inference helpers used by the app
│   └── plotting.py
├── app/streamlit_app.py        # web UI
├── tests/test_pipeline.py
├── data/raw/                   # input CSV
├── models/                     # generated: .joblib models + metadata.json
├── reports/                    # generated: metrics CSVs and figures
├── notebooks/                  # original notebook, kept for reference
├── run_all.py                  # one-command workflow
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (required once; trained models are not stored in the repo)
python -m anxiety.train_regression

# Launch the app
streamlit run app/streamlit_app.py
```

To also run the EDA and the classifier, use `python run_all.py` instead of the training line.

Run steps individually:

```bash
python -m anxiety.eda
python -m anxiety.train_regression
python -m anxiety.train_classification
python -m anxiety.predict          # sanity check on one example row
pytest
```

Run everything from the project root so `anxiety` is importable.

The zip ships with the trained regressor so the app works immediately; the classifier
(~120 MB) is not included and is recreated by `python -m anxiety.train_classification`.
The ngrok token in the notebook copy has been redacted.

## Design notes

- **One pipeline, saved whole.** Feature engineering and IQR clipping are part of the
  saved model, so the app passes raw inputs straight in. The old app re-implemented
  both steps by hand, which risks train/serve drift.
- **No preprocessing leakage.** IQR bounds, imputer medians and scaler statistics are
  learned on the training split only. The notebook clipped and imputed the full
  dataset before splitting, so test metrics here can differ slightly.
- **Best model is picked automatically** by 5-fold CV RMSE rather than hard-coded.
- **Severity bands** (`SEVERITY_BINS` / `SEVERITY_LABELS` in `config.py`) are shared by
  the classifier target and the app. The notebook's app called the second band
  "Medium"; it is "Moderate" everywhere now.
- The app reads dropdown options and ranges from `models/metadata.json`, written during training.

## Results from the included run

| Task | Model | Hold-out |
|---|---|---|
| Regression | Gradient Boosting (tuned) | RMSE 1.00, MAE 0.81, R² 0.78 |
| Classification | Random Forest | accuracy 0.75, macro-F1 0.60 |

The classifier does poorly on the "High" band (about 4% of rows; recall around 0.06),
so macro-F1 is far below accuracy. The macro-F1 gain over logistic regression is not
distinguishable from zero (bootstrap CI includes 0).

## Disclaimer

This is an educational ML project, not a diagnostic tool.
