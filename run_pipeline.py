"""
run_pipeline.py
----------------
فایل جدید — طبق مستند پروژه، باید یک فایل اصلی وجود داشته باشد که کل
فرآیند (بارگذاری داده تا ثبت در MLflow) را به‌صورت خودکار اجرا کند.
قبلاً هر مرحله را باید دستی (preprocessing.py، سپس features.py، سپس
train.py) اجرا می‌کردید؛ حالا با یک دستور کل چرخه اجرا می‌شود:

    python run_pipeline.py
"""

import os
import json
import joblib
import pandas as pd

from src.preprocessing import preprocess
from src.features import create_features
from src.train import split_data, scale_numeric, train_all_models, get_best_model, SEED
from src.evaluate import evaluate_model
from src.mlflow_utils import init_mlflow, log_run

RESULTS_DIR = "results"


def run_for_version(df: pd.DataFrame, dataset_version: str, seed: int = SEED):
    print(f"\n{'='*70}\n نسخه دیتاست: {dataset_version} | شکل: {df.shape}\n{'='*70}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=seed)
    X_train, X_val, X_test, scaler = scale_numeric(X_train, X_val, X_test)

    results = train_all_models(X_train, y_train, X_val, y_val, seed=seed)

    rows = []
    for model_name, res in results.items():
        model = res["model"]
        eval_out = evaluate_model(model, X_test, y_test)
        metrics, cm = eval_out["metrics"], eval_out["confusion_matrix"]

        log_run(
            model_name=model_name,
            dataset_version=dataset_version,
            params={k: v for k, v in res["params"].items() if isinstance(v, (int, float, str, bool)) or v is None},
            metrics=metrics,
            seed=seed,
            confusion_matrix=cm,
            model=model,
        )

        rows.append({
            "dataset_version": dataset_version,
            "model": model_name,
            "cv_f1_mean": res["cv_scores"].mean(),
            "val_f1": res["val_f1"],
            **{f"test_{k}": v for k, v in metrics.items()},
        })

    best_name, best_res = get_best_model(results)
    return rows, (best_name, best_res, X_test, y_test, scaler)


def main():
    init_mlflow()

    # مرحله ۲: پاکسازی -> v2
    df_v2 = preprocess()

    # مرحله ۳: مهندسی ویژگی -> v3
    df_v3 = create_features()

    # مرحله ۴+۵+۶: آموزش، ارزیابی و ثبت در MLflow برای هر دو نسخه
    all_rows, candidates = [], {}
    for version_name, df in [("v2", df_v2), ("v3", df_v3)]:
        rows, best_candidate = run_for_version(df, version_name)
        all_rows.extend(rows)
        candidates[version_name] = best_candidate

    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison_df = pd.DataFrame(all_rows).sort_values("test_f1_score", ascending=False)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\n" + comparison_df.to_string(index=False))

    best_row = comparison_df.iloc[0]
    best_version = best_row["dataset_version"]
    _, best_res, X_test, y_test, scaler = candidates[best_version]

    joblib.dump(
        {"model": best_res["model"], "scaler": scaler, "feature_columns": list(X_test.columns)},
        "models/best_model.pkl",
    )
    with open(os.path.join(RESULTS_DIR, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_version": best_version,
            "model_name": best_row["model"],
            "test_f1_score": best_row["test_f1_score"],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ بهترین مدل: {best_row['model']} روی {best_version} -> models/best_model.pkl")


if __name__ == "__main__":
    main()
