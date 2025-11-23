# Heart Attack Analysis & Prediction — Mini Project

This repo explores the Heart Attack Analysis & Prediction (HAAP) dataset and builds a simple, effective baseline using scikit-learn. It includes EDA, preprocessing pipelines, cross-validated models, metrics, and saved artifacts.

## Project structure

- `heart.csv` — dataset (root)
- `o2Saturation.csv` — optional single-column series that can be merged if length matches
- `notebooks/01_eda.ipynb` — end-to-end EDA + baseline modeling + artifacts saving
- `artifacts/` — created by the notebook (model, metrics, split, summary)
- `requirements.txt` — Python dependencies
- `project process log.md` — running changelog of steps and findings

## Environment setup (Windows, PowerShell)

A virtual environment is already configured at `.venv` for this workspace.

Optional re-create (only if desired):

1) Create and activate venv
2) Install deps from requirements.txt

## How to run

1) Open `notebooks/01_eda.ipynb` in VS Code and select the Python interpreter from `.venv`.
2) Run cells top-to-bottom. The notebook will:
   - Validate and load `heart.csv` (and optionally merge `o2Saturation.csv`)
   - Perform EDA and target balance checks
   - Build preprocessing (scaling + one-hot)
   - Train and evaluate Logistic Regression and a tree model (RandomForest or XGBoost)
   - Tune hyperparameters (small safe grids)
   - Evaluate on a held-out test set (metrics + curves)
   - Save artifacts into `./artifacts`
   - Append a short progress note to `project process log.md`

### Batch scoring (CLI)

Score new data files without opening the notebook. Input can be CSV (with the same columns as training, excluding `output`) or JSON (array of objects).

PowerShell examples:

```powershell
# Score a CSV and save predictions
python src/score.py --input data/new_patients.csv --output predictions.csv

# Use a custom decision threshold
python src/score.py --input data/new_patients.csv --threshold 0.42 --output predictions.csv

# Score a JSON file and print a preview
python src/score.py --input data/sample.json
```

Predictions include:

- `proba`: probability of heart attack (0..1)
- `pred`: decision (0/1) based on a threshold (CLI arg > cost-based > Youden’s J > 0.5)

## Notes

- GPU is optional. If `nvidia-smi` and `xgboost` are available, the notebook will prefer GPU via `gpu_hist`.
- You can tweak feature lists, grids, and plots as needed.
- For a short video, screenshots of EDA and ROC/PR curves plus a quick narrative of metrics should suffice.

## Artifacts for reporting

Saved under `./artifacts` after running the notebook:

- `model_pipeline.joblib`, `metrics_test.json`, `split.json`, `artifacts_summary.json`
- `roc_curve.png/.csv`, `pr_curve.png/.csv`
- `confusion_matrix_0_5.png`, `confusion_matrix_thr_<thr>.png`
- `calibration_curve.png/.csv`, `calibration.json`
- `threshold_cost_curve.csv`, `threshold_cost_selection.json`
- `permutation_importance.csv/.png`

## Next ideas (optional)

- Add calibration with `CalibratedClassifierCV`
- Try other models (LightGBM/CatBoost) if allowed
- Add k-fold cross-validation with repeated runs for stability
- Export a minimal scoring script under `src/` for batch inference
