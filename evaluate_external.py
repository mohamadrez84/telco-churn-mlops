"""Evaluate the existing model on a separate, labeled cohort."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.features import prepare_raw_features
from src.preprocessing import RAW_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
CUSTOMER_ID = "CustomerID"
TARGET = "Churn Value"


def evaluate_external(cohort, reference_ids, bundle):
    required = RAW_COLUMNS | {CUSTOMER_ID, TARGET}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    ids = cohort[CUSTOMER_ID].astype("string").str.strip().str.casefold()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be present and unique")
    reference = reference_ids.astype("string").str.strip().str.casefold()
    overlap = int(ids.isin(reference).sum())
    if overlap:
        raise ValueError(f"{overlap} customer IDs overlap with the IBM reference dataset")

    labels = pd.to_numeric(cohort[TARGET], errors="raise")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("Churn Value must contain both classes and only 0 or 1")
    X = prepare_raw_features(cohort[sorted(RAW_COLUMNS)], bundle)
    result = evaluate_model(bundle["model"], X, labels.astype(int))
    return {
        "model_name": type(bundle["model"]).__name__,
        "dataset_version": bundle["dataset_version"],
        "n_customers": len(cohort),
        "n_churn": int(labels.sum()),
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="Independent CSV or .xlsx with raw inputs and labels")
    args = parser.parse_args()
    try:
        if args.cohort.suffix.lower() == ".xlsx":
            cohort = pd.read_excel(args.cohort, dtype={CUSTOMER_ID: "string"})
        elif args.cohort.suffix.lower() == ".csv":
            cohort = pd.read_csv(args.cohort, dtype={CUSTOMER_ID: "string"})
        else:
            raise ValueError("Expected a .csv or .xlsx file")
        bundle = joblib.load(BASE_DIR / "models" / "best_model.pkl")
        result = evaluate_external(cohort, load_data()[CUSTOMER_ID], bundle)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
