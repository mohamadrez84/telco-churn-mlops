"""Tune candidates on train and validation, then evaluate the winner on test."""
import json
from hashlib import file_digest
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import RAW_XLSX_PATH
from src.preprocessing import preprocess
from src.features import create_features, transform_features
from src.train import (
    DECISION_THRESHOLDS, OOF_F1_TOLERANCE, PARAMETER_GRID, SEED, VALIDATION_F1_TOLERANCE,
    select_candidates, split_data, train_all_models,
)
from src.evaluate import evaluate_model
from src.mlflow_utils import init_mlflow, log_run, save_confusion_matrix

BASE_DIR = Path(__file__).resolve().parent


def main():
    init_mlflow()
    df_v2 = preprocess()
    df_v3 = create_features(df_v2)

    groups = {}
    test_sets = {}
    for version, df in [("v2", df_v2), ("v3", df_v3)]:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        test_sets[version] = (X_test, y_test)
        for name, options in train_all_models(X_train, y_train, X_val, y_val, version).items():
            groups[version, name] = options

    candidates, best_key, f1_floor = select_candidates(groups)
    best_version, best_name = best_key
    best = candidates[best_key]
    X_test, y_test = test_sets[best_version]
    test = transform_features(X_test, best["scaler"], best["high_value_threshold"])
    evaluation = evaluate_model(best["model"], test, y_test, best["decision_threshold"])

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = results_dir / "confusion_matrix.png"
    save_confusion_matrix(evaluation["confusion_matrix"], matrix_path, f"{best_name} on {best_version}")

    rows = []
    experiments = []
    for (version, name), result in candidates.items():
        selected = (version, name) == best_key
        bundle = {
            key: result[key]
            for key in ("model", "scaler", "feature_columns", "high_value_threshold", "decision_threshold")
        }
        bundle["dataset_version"] = version
        metrics = {
            "cv_f1_mean": float(result["cv_scores"].mean()),
            "cv_f1_std": float(result["cv_scores"].std()),
            "val_f1": float(result["val_f1"]),
            "val_accuracy": float(result["val_accuracy"]),
            "baseline_val_f1": float(groups[version, name][0]["val_f1"]),
            "baseline_val_accuracy": float(groups[version, name][0]["val_accuracy"]),
            "decision_threshold": result["decision_threshold"],
        }
        if selected:
            metrics.update({f"test_{key}": value for key, value in evaluation["metrics"].items()})
        params = result["model"].get_params()
        log_run(
            name, version, params, metrics, SEED, bundle,
            confusion_matrix_path=matrix_path if selected else None,
        )
        experiments.append({
            "dataset_version": version,
            "model": name,
            "selected": selected,
            "params": {key: str(value) for key, value in sorted(params.items())},
            "metrics": metrics,
        })
        rows.append({"dataset_version": version, "model": name, "selected": selected, **metrics})
        if selected:
            best_bundle = bundle
            best_info = {"model_name": name, "dataset_version": version, **metrics}

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows).sort_values(["selected", "val_accuracy", "val_f1"], ascending=False)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    joblib.dump(best_bundle, models_dir / "best_model.pkl")
    (results_dir / "best_model_info.json").write_text(
        json.dumps(best_info, indent=2) + "\n", encoding="utf-8",
    )
    with RAW_XLSX_PATH.open("rb") as dataset:
        dataset_hash = file_digest(dataset, "sha256").hexdigest()
    snapshot = {
        "seed": SEED,
        "source_data_sha256": dataset_hash,
        "selection_metric": "val_accuracy",
        "validation_f1_floor": f1_floor,
        "validation_f1_tolerance": VALIDATION_F1_TOLERANCE,
        "oof_f1_tolerance": OOF_F1_TOLERANCE,
        "threshold_grid": DECISION_THRESHOLDS.tolist(),
        "parameter_grid": PARAMETER_GRID,
        "experiments": experiments,
    }
    (results_dir / "experiment_summary.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Selected {best_name} on {best_version} by validation accuracy (F1 floor: {f1_floor:.4f})")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
