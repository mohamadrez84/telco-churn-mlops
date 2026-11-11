import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

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


def train_all_models(X_train, y_train, X_val, y_val, dataset_version, seed=SEED):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_train, fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        scaler, threshold = fit_transforms(fold_train, dataset_version)
        folds.append((
            transform_features(fold_train, scaler, threshold), y_train.iloc[train_idx],
            transform_features(fold_val, scaler, threshold), y_train.iloc[val_idx],
        ))

    scaler, threshold = fit_transforms(X_train, dataset_version)
    train = transform_features(X_train, scaler, threshold)
    val = transform_features(X_val, scaler, threshold)
    results = {}
    for name, builder in MODEL_REGISTRY.items():
        cv_scores = np.array([
            f1_score(y_fold_val, fit_model(name, builder, X_fold, y_fold, seed).predict(X_fold_val))
            for X_fold, y_fold, X_fold_val, y_fold_val in folds
        ])
        model = fit_model(name, builder, train, y_train, seed)
        val_f1 = f1_score(y_val, model.predict(val))
        print(f"[train] {name} | CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Validation F1: {val_f1:.4f}")
        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "val_f1": val_f1,
            "scaler": scaler,
            "high_value_threshold": threshold,
            "feature_columns": list(train.columns),
        }
    return results
