import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.features import transform_features

TARGET_COLUMN = "Churn Value"
SEED = 42
N_FOLDS = 5

NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges", "AvgChargePerMonth"]

MODEL_REGISTRY = {
    "LogisticRegression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    "RandomForest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=4),
    "XGBoost": lambda seed: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=seed, n_jobs=4),
    "CatBoost": lambda seed: CatBoostClassifier(iterations=200, random_state=seed, verbose=0, auto_class_weights="Balanced", thread_count=4, allow_writing_files=False),
}


PARAMETER_GRID = {
    "LogisticRegression": ({}, {"C": 0.05}, {"C": 0.2}, {"C": 5}, {"class_weight": None}),
    "RandomForest": (
        {}, {"min_samples_leaf": 5}, {"min_samples_leaf": 12},
        {"max_depth": 8, "min_samples_leaf": 8},
        {"max_features": 0.5, "min_samples_leaf": 10},
    ),
    "XGBoost": (
        {},
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        {"max_depth": 2, "learning_rate": 0.06, "min_child_weight": 4},
        {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5, "reg_lambda": 5},
    ),
    "CatBoost": (
        {}, {"depth": 4, "learning_rate": 0.05},
        {"depth": 5, "learning_rate": 0.06, "l2_leaf_reg": 7},
        {"depth": 3, "iterations": 350, "learning_rate": 0.06},
        {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 8},
    ),
}
DECISION_THRESHOLDS = np.arange(24, 81) / 100
OOF_F1_TOLERANCE = 0.015
VALIDATION_F1_TOLERANCE = 0.01


def split_data(df: pd.DataFrame, seed: int = SEED):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=seed
    )
    print(f"[train] train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_transforms(X, dataset_version):
    threshold = float(X["Monthly Charges"].median()) if dataset_version == "v3" else None
    columns = [column for column in NUMERIC_COLUMNS if column in X]
    return StandardScaler().fit(X[columns]), threshold


def fit_model(name, builder, X, y, seed):
    model = builder(seed)
    if name == "XGBoost":
        negative, positive = np.bincount(y)
        model.set_params(scale_pos_weight=negative / positive)
    return model.fit(X, y)


def threshold_candidates(y, probabilities):
    f1_scores = np.array([f1_score(y, probabilities >= t) for t in DECISION_THRESHOLDS])
    best_f1 = int(np.argmax(f1_scores))
    eligible = np.flatnonzero(f1_scores >= f1_scores[best_f1] - OOF_F1_TOLERANCE)
    best_accuracy = max(eligible, key=lambda i: accuracy_score(y, probabilities >= DECISION_THRESHOLDS[i]))
    return tuple(dict.fromkeys((0.5, float(DECISION_THRESHOLDS[best_f1]), float(DECISION_THRESHOLDS[best_accuracy]))))


def train_all_models(X_train, y_train, X_val, y_val, dataset_version, seed=SEED):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_train, fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        scaler, threshold = fit_transforms(fold_train, dataset_version)
        folds.append((
            transform_features(fold_train, scaler, threshold), y_train.iloc[train_idx],
            transform_features(fold_val, scaler, threshold), val_idx,
        ))

    scaler, high_value_threshold = fit_transforms(X_train, dataset_version)
    train = transform_features(X_train, scaler, high_value_threshold)
    val = transform_features(X_val, scaler, high_value_threshold)
    results = {}
    for name, builder in MODEL_REGISTRY.items():
        candidates = []
        for params in PARAMETER_GRID[name]:
            def configured(seed):
                return builder(seed).set_params(**params)

            oof = np.empty(len(y_train))
            for X_fold, y_fold, X_fold_val, val_idx in folds:
                oof[val_idx] = fit_model(name, configured, X_fold, y_fold, seed).predict_proba(X_fold_val)[:, 1]

            model = fit_model(name, configured, train, y_train, seed)
            probabilities = model.predict_proba(val)[:, 1]
            for threshold in threshold_candidates(y_train, oof):
                predicted = probabilities >= threshold
                cv_scores = np.array([
                    f1_score(y_train.iloc[idx], oof[idx] >= threshold) for _, _, _, idx in folds
                ])
                candidates.append({
                    "model": model,
                    "cv_scores": cv_scores,
                    "val_f1": f1_score(y_val, predicted),
                    "val_accuracy": accuracy_score(y_val, predicted),
                    "decision_threshold": threshold,
                    "scaler": scaler,
                    "high_value_threshold": high_value_threshold,
                    "feature_columns": list(train.columns),
                })
        print(f"[train] {name} | {len(candidates)} settings and thresholds compared on validation")
        results[name] = candidates
    return results


def select_candidates(groups):
    # Each group's first entry is the default estimator at a 0.5 threshold.
    baseline_f1 = max(group[0]["val_f1"] for group in groups.values())
    f1_floor = baseline_f1 - VALIDATION_F1_TOLERANCE
    selected = {}
    for key, group in groups.items():
        eligible = [candidate for candidate in group if candidate["val_f1"] >= f1_floor]
        if eligible:
            selected[key] = max(eligible, key=lambda row: (row["val_accuracy"], row["val_f1"]))
        else:
            selected[key] = max(group, key=lambda row: (row["val_f1"], row["val_accuracy"]))

    best_key = max(
        (key for key, row in selected.items() if row["val_f1"] >= f1_floor),
        key=lambda key: (selected[key]["val_accuracy"], selected[key]["val_f1"]),
    )
    return selected, best_key, f1_floor
