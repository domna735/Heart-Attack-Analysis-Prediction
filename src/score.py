#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI scorer for the saved model pipeline.

Usage examples (PowerShell):

# Score a CSV file and write predictions to a new CSV
python src/score.py --input data/new_patients.csv --output predictions.csv

# Score a JSON file and echo results
python src/score.py --input data/sample.json

# Provide a custom decision threshold for positive class
python src/score.py --input data/new_patients.csv --threshold 0.42 --output predictions.csv

Inputs:
- CSV: header row with same columns as training (except 'output')
- JSON: JSON array of objects with the same schema

Outputs:
- When --output is provided for CSV input: writes a CSV with columns: original + proba + pred
- Otherwise, prints a preview to stdout
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "artifacts_summary.json"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_threshold(arg_threshold: float | None) -> float:
    # Priority: CLI arg > cost-based threshold > opt_threshold > 0.5
    if arg_threshold is not None:
        return float(arg_threshold)
    thr = None
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                s = json.load(f)
            thr = s.get("cost_based_threshold") or s.get("opt_threshold")
        except Exception:
            thr = None
    return float(thr) if thr is not None else 0.5


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        sys.exit("Unsupported input format. Use .csv or .json")
    # Drop target column if present
    if "output" in df.columns:
        df = df.drop(columns=["output"])
    return df


def make_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    model = load_model()
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["proba"] = proba
    out["pred"] = pred
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score new data with saved model pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV or JSON")
    parser.add_argument("--output", help="Optional path to save predictions CSV")
    parser.add_argument("--threshold", type=float, help="Decision threshold (default: from artifacts_summary or 0.5)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    df = read_input(input_path)
    threshold = load_threshold(args.threshold)

    preds = make_predictions(df, threshold)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)
        print(f"Predictions saved to: {out_path}")
    else:
        # Print small preview
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(preds.head(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
