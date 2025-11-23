# Heart Attack Analysis & Prediction (HAAP) — Final Report

Author: Your Name
Date: 2025-09-13

---

## 1. Problem Formulation

- Objective: Predict risk of heart attack (binary outcome) from clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, ECG results, heart rate, etc.).
- Dataset: `heart.csv` (303 rows, 14 columns, target column `output`). Target prevalence ≈ 0.545.
- Business framing: Provide a probability score (0–1) per patient and a configurable decision threshold to trade off recall vs precision depending on clinical priorities.
- Success criteria:
  - Technical: AUC-ROC and AUC-PR on a held-out test set; well-calibrated probabilities (low Brier score).
  - Practical: Clear, reproducible pipeline; threshold recommendations to support different clinical priorities.

## 2. System Design & Implementation

 
### 2.1 Data and Features

- Source: Kaggle HAAP dataset.
- Target: `output` (0/1).
- Numeric features: `age, trtbps, chol, thalachh, oldpeak`.
- Categorical features: `sex, cp, fbs, restecg, exng, slp, caa, thall`.
- Optional external signal `o2Saturation.csv` detected but not merged (length mismatch with heart.csv).

### 2.2 Preprocessing

- ColumnTransformer pipeline:
  - StandardScaler for numeric.
  - OneHotEncoder(handle_unknown='ignore') for categorical.
- Train/test split: stratified 80/20, fixed random_state=42; indices persisted to `artifacts/split.json`.

### 2.3 Models and Training

- Baseline: DummyClassifier (most frequent).
- Primary models: Logistic Regression (balanced class_weight) and RandomForest (or XGBoost if available).
- Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
- Hyperparameter tuning: GridSearchCV on both families, select best by CV ROC AUC.
- Best model selected: LogisticRegression.

### 2.4 Evaluation and Artifacts

- Test metrics (from `artifacts/metrics_test.json`):
  - Accuracy: 0.8525
  - Precision: 0.8529
  - Recall: 0.8788
  - F1: 0.8657
  - ROC AUC: 0.9037
  - PR AUC: 0.9224
- Plots and CSVs saved to `artifacts/`:
  - ROC (`roc_curve.png/.csv`), PR (`pr_curve.png/.csv`), Confusion matrices (`confusion_matrix_0_5.png`, `confusion_matrix_thr_<thr>.png`).
  - Calibration (`calibration_curve.png/.csv`), Brier score in `calibration.json` (0.1279).
  - Permutation importance (`permutation_importance.csv/.png`).
- Pipeline persisted as `model_pipeline.joblib` for scoring.

### 2.5 Threshold Selection

- Youden’s J threshold (from `artifacts_summary.json`): 0.0351.
- Cost-based threshold (from `threshold_cost_selection.json`): 0.45 with cost weights C_FN=5, C_FP=1.
  - Metrics at cost-based threshold: precision=0.838, recall=0.939, F1=0.886, confusion counts tp=31, fp=6, tn=22, fn=2.
- Guidance:
  - If minimizing missed positives is critical, prefer the cost-based (higher recall) threshold.
  - Otherwise, default 0.5 or the Youden threshold for balanced trade-offs.

### 2.6 Explainability

- Permutation importance computed on the full pipeline to ensure end-to-end correctness. Results saved to CSV/PNG.
- For logistic regression, coefficient inspection possible; SHAP attempted if installed.

### 2.7 Reproducibility

- Environment: Python 3.13.7 in `.venv`.
- Requirements pinned in `requirements.txt`.
- Split indices saved; RNG seeds fixed.
- Artifacts and intermediate CSVs/JSONs recorded consistently under `artifacts/`.

## 3. Innovation and Improvements

- Cost-sensitive thresholding: Introduced a cost model (C_FN vs C_FP) to select thresholds aligned to clinical priorities, with a full threshold-cost curve for transparency.
- Calibration analysis: Added reliability diagram and Brier score to verify probability quality, not just ranking metrics.
- Robust artifact workflow: All key figures and CSVs persisted; logging cell and artifact cell resilient to kernel restarts (auto-recompute or load thresholds).
- Explainability: Model-agnostic permutation importance on the full pipeline provides actionable insights regardless of model choice.
- Scoring CLI: `src/score.py` enables non-notebook batch inference (CSV/JSON), with optional custom threshold.
- GPU-ready option: Opportunistic use of XGBoost with GPU when available.

## 4. Results, Analysis, and Discussion

- Performance is strong for a small dataset (n=303), with ROC AUC ≈ 0.904 and PR AUC ≈ 0.922 on the held-out test set.
- Calibration is reasonable (Brier ≈ 0.128), indicating usable probability estimates.
- Thresholds:
  - Youden’s J ≈ 0.035 suggests the model outputs probabilities skewed low—common with regularized linear models and scaling.
  - A more practical operating point may be the cost-based threshold (0.45), yielding high recall (≈0.939) and solid precision (≈0.838).
- Limitations:
  - Small sample size; estimates have variance. Prefer cross-validated reporting or repeated CV for publication.
  - Demographics and clinical context: external validity not guaranteed without further evaluation.

## 5. Implementation Details (Tools/Codes Used)

- Language/Env: Python 3.13.7, Jupyter.
- Core libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, shap, xgboost (optional).
- Notebook: `notebooks/01_eda.ipynb` — end-to-end EDA, training, evaluation, artifacts.
- CLI: `src/score.py` — batch scoring with threshold support.
- Artifacts: saved to `artifacts/` (see list above).
- System overview: `docs/SYSTEM_OVERVIEW.md` — non-technical explanation and usage.

## 6. How to Reproduce

1. Create/activate the project `.venv` in VS Code (already present).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Open and run `notebooks/01_eda.ipynb` sequentially or run the key cells:
   - Data load, training, evaluation, artifact saving.
4. To score new data:
   - `python src/score.py --input data/sample_patients.csv --output predictions_sample.csv`

## 7. Figures and Tables for the Report

Include from `artifacts/`:

- ROC and PR curves (PNG + CSV).
- Confusion matrix at default and selected thresholds.
- Calibration curve and Brier score.
- Permutation importance bar chart.
- Table: `metrics_test.json` values formatted.

## 8. Conclusion and Next Steps

- The solution meets the objectives: clear formulation, reproducible system, and strong baseline performance with explainability and cost-aware decisions.
- Next improvements: larger datasets, external validation, additional models (e.g., gradient boosting), and domain-driven feature engineering.

---

Appendix: Dataset schema is documented in the project target; features are validated in the notebook. All steps and artifacts are traced in `project process log.md`.
