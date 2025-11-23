# System Overview — Heart Attack Analysis & Prediction

This document explains the full system end-to-end in plain terms, suitable for readers without a technical background. It also documents the tools and public resources used to build the solution.

## What the system does

- Takes a dataset of patient health measurements (age, blood pressure, cholesterol, etc.).
- Learns patterns that distinguish patients who are more likely vs less likely to have a heart attack.
- Produces a trained model that can score new patients with a probability between 0 and 1.
- Allows choosing a “decision threshold” to convert that probability into a Yes/No decision, depending on what matters more: catching more positives (higher recall) or avoiding false alarms (higher precision).
- Provides charts and files to evaluate performance and understand which features matter the most.

## What’s included

- A Jupyter Notebook (`notebooks/01_eda.ipynb`) that:
  - Loads and checks the data.
  - Explores the dataset (EDA) to understand ranges, missing values, and feature types.
  - Preprocesses the data (scales numeric features, one-hot encodes categories).
  - Trains two reference models (Logistic Regression and RandomForest) with cross-validation.
  - Tunes a few parameters and picks the best model by AUC (a standard metric for binary classification).
  - Evaluates the best model on a held-out test set.
  - Saves the trained model and metrics into the `artifacts/` folder.
  - Generates and saves plots: ROC, Precision-Recall, Confusion Matrix, Calibration.
  - Computes an “optimal” threshold by a rule (Youden’s J) and an alternative threshold using a simple cost model.
  - Saves explainability artifacts (permutation importance) to show which features drive predictions.

- A ready-to-use script (`src/score.py`) to score new data files (CSV or JSON) without opening the notebook.

- A process log (`project process log.md`) that records actions, results, and decisions in a consistent template.

## Key outputs (artifacts)

All important results are saved into the `artifacts/` folder:

- `model_pipeline.joblib` — The trained model plus preprocessing steps.
- `metrics_test.json` — Final performance on the test set (accuracy, precision, recall, F1, ROC AUC, PR AUC).
- `split.json` — Which rows were used for training vs testing (reproducibility).
- `artifacts_summary.json` — Summary including best model name and chosen thresholds.
- Plots and data for reporting:
  - `roc_curve.png/.csv`, `pr_curve.png/.csv` — ROC and Precision-Recall curves.
  - `confusion_matrix_0_5.png` and `confusion_matrix_thr_<thr>.png` — Confusion matrices at common thresholds.
  - `calibration_curve.png/.csv`, `calibration.json` — Reliability of probabilities and Brier score.
  - `threshold_cost_curve.csv`, `threshold_cost_selection.json` — Cost-based threshold exploration and choice.
  - `permutation_importance.csv/.png` — Model-agnostic feature importance results.

## How to use the scoring script (no coding required)

1) Prepare a CSV file with the same columns as the training data (except the target column `output`).
2) Open PowerShell in the project folder.
3) Run the script:

```powershell
python src/score.py --input data/new_patients.csv --output predictions.csv
```

- This writes a new file `predictions.csv` with two added columns:
  - `proba`: the probability of heart attack (0 to 1)
  - `pred`: the decision (0 or 1) using the default threshold

Optional: Set a custom threshold (for example, to catch more positives):

```powershell
python src/score.py --input data/new_patients.csv --threshold 0.40 --output predictions.csv
```

- If you omit `--output`, the script prints the first few results in the terminal.

## Choosing a threshold (simple guidance)

- Default decision threshold is 0.5 unless the summary file provides a better one.
- The notebook saves two alternatives:
  - “Youden’s J” threshold: balances sensitivity and specificity.
  - “Cost-based” threshold: you can set a higher cost for missing a true case (false negative) than for raising a false alarm (false positive). The system searches thresholds to minimize this cost.
- For health screening, it’s common to prefer higher recall (catching more true cases). That usually means choosing a lower threshold.

## What public tools we used (with links)

- Python 3, Jupyter Notebook — interactive data analysis ([jupyter.org](https://jupyter.org/))
- pandas — data loading and manipulation ([pandas.pydata.org](https://pandas.pydata.org/))
- numpy — numerical computing ([numpy.org](https://numpy.org/))
- scikit-learn — modeling, metrics, pipelines, calibration, permutation importance ([scikit-learn.org](https://scikit-learn.org/stable/))
- matplotlib, seaborn — plots ([matplotlib.org](https://matplotlib.org/) , [seaborn.pydata.org](https://seaborn.pydata.org/))

## Why we chose these tools

- They are widely used, well-documented, and stable for tabular ML.
- scikit-learn Pipelines make it easy to save the full process (preprocessing + model) so new data can be scored consistently later.
- The approach is transparent and explainable, which is important for healthcare-related tasks.

## What to hand in (mapped to your assignment)

- Codes and how we built it:
  - Notebook: `notebooks/01_eda.ipynb` contains the full workflow and comments.
  - Script: `src/score.py` for batch scoring without notebooks.
  - Requirements: `requirements.txt` lists the Python packages.

- Final report materials:
  - Use figures and tables from `artifacts/` (ROC/PR, confusion matrices, calibration, permutation importance, metrics JSON).
  - The process log (`project process log.md`) summarizes steps, results, and decisions to help write the report narrative.

- Optional short video suggestions:
  - Show a few plots (ROC, PR, confusion matrix), explain the meaning in plain language.
  - Demonstrate running `src/score.py` on a tiny CSV and show the resulting `predictions.csv`.

## Notes and caveats

- The dataset is relatively small (303 rows). For more robust results, consider cross-validating with repeated folds or obtaining more data.
- This is a baseline; other models (e.g., gradient boosting) might perform similarly or better. Focus here is clarity and reproducibility.
- The model is not a medical device; it’s for educational purposes and exploratory analysis only.

## Glossary (plain-English)

- Accuracy: Overall fraction of correct decisions.
- Precision: Of all predicted positives, how many were truly positive.
- Recall (Sensitivity): Of all true positives, how many did we catch.
- F1 score: A single number balancing precision and recall.
- ROC AUC: A measure of ranking quality across thresholds (higher is better).
- PR AUC: Summarizes the precision–recall tradeoff; useful for imbalanced data.
- Calibration: Whether predicted probabilities reflect real-world frequencies (e.g., among people with 0.7 predicted risk, ~70% are truly positive).
- Confusion matrix: Counts of TP/FP/TN/FN for a chosen threshold.
- Permutation importance: How much each feature impacts model performance when shuffled.

---

If you need a simpler or more step-by-step explanation for your report, we can create a brief “Executive Summary” page tailored to non-technical readers.
