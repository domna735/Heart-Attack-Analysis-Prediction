# Project process log

Template for entries:

## YYYY-MM-DD | Short Title

Intent:

Action:

Result:

Decision / Interpretation:

Next:

---

## 2025-09-13 | Project initialization

Intent:

- Set up a reproducible environment and scaffold an end-to-end EDA + modeling workflow.

Action:

- Created and configured a Python virtual environment for the workspace and installed core packages: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter.
- Added `requirements.txt` for reproducibility.
- Scaffolded `notebooks/01_eda.ipynb` with an end-to-end outline: environment checks, data loading/validation, EDA, preprocessing, CV models (LogReg + Tree/XGBoost), tuning, evaluation (ROC/PR, confusion matrix), threshold tuning, explainability, and artifact persistence.
- Created `README.md` with setup and run instructions.

Result:

- Workspace initialized and ready to run; notebook kernel configured to use the project virtual environment.

Decision / Interpretation:

- Start with scikit-learn baselines (LogisticRegression, RandomForest). GPU/XGBoost optional.

Next:

- Run the notebook to compute dataset summary, class balance, train models, save artifacts, and append results into this log.

## 2025-09-13 | Baseline models trained and evaluated

Intent:

- Train baseline models and evaluate on a held-out test set; save minimal artifacts.

Action:

- Loaded `heart.csv` (303 rows, 14 columns). Identified numeric and categorical features, built preprocessing (scaling + one-hot).
- Performed train/test split (train 242, test 61, stratified).
- Trained DummyClassifier baseline, Logistic Regression (with CV), and a RandomForest model; compared metrics.
- Tuned hyperparameters via GridSearchCV (small, safe grids) and selected best model by CV ROC AUC.

Result:

- Target rate p1 ≈ 0.545. Best model: LogisticRegression with CV AUC ≈ 0.897.
- Test metrics (selected model): AUC ≈ 0.904, F1 ≈ 0.866, Accuracy ≈ 0.852.
- Initial optimal threshold (Youden’s J) ≈ 0.476.
- Saved artifacts to `./artifacts`: model (`model_pipeline.joblib`), metrics (`metrics_test.json`), split (`split.json`), summary (`artifacts_summary.json`).

Decision / Interpretation:

- Logistic Regression generalizes best in CV and test; proceed using it as the reference model.

Next:

- Persist plots (ROC/PR/confusion matrix), add calibration, cost-based threshold, and explainability; then document and add a scoring CLI.

## 2025-09-13 | Plots, calibration, and thresholds saved

Intent:

- Produce reporting-ready plots and robust thresholding artifacts.

Action:

- Added resilient cells to save: ROC (`roc_curve.png`/`.csv`), PR (`pr_curve.png`/`.csv`), confusion matrices at 0.5 and at Youden’s J, calibration curve (`calibration_curve.png`/`.csv`) and Brier score (`calibration.json`).
- Implemented cost-based threshold selection (`threshold_cost_curve.csv`, `threshold_cost_selection.json`) with example costs C_FN=5 (false negatives) and C_FP=1.
- Updated `artifacts_summary.json` with `opt_threshold` and `cost_based_threshold`.

Result:

- Files in `artifacts/`: `roc_curve.png/.csv`, `pr_curve.png/.csv`, `confusion_matrix_0_5.png`, `confusion_matrix_thr_<thr>.png`, `calibration_curve.png/.csv`, `calibration.json`, `threshold_cost_curve.csv`, `threshold_cost_selection.json`.

Decision / Interpretation:

- The model performs well with ROC AUC ~0.904; calibration appears reasonable (see Brier score). If recall is a priority, the cost-based threshold can be preferred.

Next:

- Generate explainability artifacts (permutation importance) and add a scoring CLI.

## 2025-09-13 | Explainability artifacts (permutation importance)

Intent:

- Provide model-agnostic interpretability for features after preprocessing.

Action:

- Computed permutation importance (scoring=ROC AUC) over the test set using the full pipeline; saved a CSV and PNG of the top features.

Result:

- Files in `artifacts/`: `permutation_importance.csv`, `permutation_importance.png`.

Decision / Interpretation:

- Importance reflects feature influence on predictive performance; consult CSV/plot for top contributors.

Next:

- Create an easy-to-use scoring CLI for batch inference on CSV/JSON and update documentation.

## 2025-09-13 | Scoring CLI for batch inference

Intent:

- Allow non-technical users to score new data files without using notebooks.

Action:

- Added `src/score.py` which loads `artifacts/model_pipeline.joblib`, reads CSV/JSON, computes probabilities and labels using a decision threshold (CLI arg > cost-based > opt > 0.5), and writes a CSV or prints a preview.

Result:

- Run via PowerShell, e.g.:
  - `python src/score.py --input data/new_patients.csv --output predictions.csv`.
  - `python src/score.py --input data/sample.json --threshold 0.42`.

Decision / Interpretation:

- This provides a simple, scriptable path to apply the trained model to new datasets.

Next:

- Update README with CLI usage and, if desired, remove unused `imbalanced-learn` from requirements.





### 2025-09-13 22:54 — Notebook progress
Data loaded: 303 rows, 14 columns.
Target rate p1: 0.545.
Split: train 242, test 61 (stratified).
Best model: LogisticRegression with CV AUC=0.897.
Test metrics: AUC=0.904, F1=0.866, Acc=0.852.
Optimal threshold (Youden's J): 0.035.
Cost-based threshold: 0.450 (C_FN=5.0, C_FP=1.0).
Artifacts saved to ./artifacts (model, metrics, split, summary).
