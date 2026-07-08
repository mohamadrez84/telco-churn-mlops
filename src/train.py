"""
train.py
--------
بازنویسی کامل نسبت به نسخه قبلی شما. تغییرات اصلی:

1) قبلاً فقط train/test (80/20) داشتید و بهترین مدل را بر اساس عملکرد
   روی همان Test انتخاب می‌کردید (اشکال روش‌شناسی: Test نباید در تصمیم‌گیری
   دخالت کند). حالا یک تفکیک سه‌گانه داریم:
       Train (70%) -> برای آموزش و Cross Validation
       Validation (15%) -> فقط برای انتخاب بهترین مدل
       Test (15%) -> فقط یک‌بار، در انتها، برای گزارش نهایی

2) قبلاً فقط v3 را آموزش می‌دادید. حالا هر دو نسخه v2 و v3 به‌طور مستقل
   آموزش/ارزیابی می‌شوند تا واقعاً ببینید مهندسی ویژگی کمک کرده یا نه.

3) نرمال‌سازی (StandardScaler) اضافه شد — فقط با fit روی Train (نه کل داده)
   تا نشتی داده در نرمال‌سازی رخ ندهد.

4) class_weight/scale_pos_weight اضافه شد چون داده نامتوازن است (~27% ریزش).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_COLUMN = "Churn Value"
SEED = 42
N_FOLDS = 5

NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges", "AvgChargePerMonth"]

MODEL_REGISTRY = {
    "LogisticRegression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    "RandomForest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1),
    "XGBoost": lambda seed: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=seed, n_jobs=-1),
    "CatBoost": lambda seed: CatBoostClassifier(iterations=200, random_state=seed, verbose=0, auto_class_weights="Balanced"),
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


def scale_numeric(X_train, X_val, X_test):
    cols = [c for c in NUMERIC_COLUMNS if c in X_train.columns]
    scaler = StandardScaler()

    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])   # fit فقط روی train
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_val, X_test, scaler


def train_all_models(X_train, y_train, X_val, y_val, seed: int = SEED):
    results = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    for name, builder in MODEL_REGISTRY.items():
        model = builder(seed)
        if name == "XGBoost":
            model.set_params(scale_pos_weight=scale_pos_weight)

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1",
                                     n_jobs=-1 if name != "CatBoost" else 1)
        print(f"[train] {name} | CV F1 (5-fold): mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val))
        print(f"[train] {name} | Validation F1: {val_f1:.4f}")

        results[name] = {"model": model, "cv_scores": cv_scores, "val_f1": val_f1, "params": model.get_params()}

    return results


def get_best_model(results: dict):
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    print(f"[train] بهترین مدل بر اساس Validation: {best_name} ({results[best_name]['val_f1']:.4f})")
    return best_name, results[best_name]
