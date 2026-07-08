"""
evaluate.py
-----------
فایل جدید — قبلاً محاسبه متریک‌ها داخل train.py قاطی بود؛ طبق ساختار
خواسته‌شده در مستند پروژه، این باید یک ماژول مجزا باشد.

مسئولیت: محاسبه معیارهای ارزیابی روی داده Test (فقط یک‌بار در پایان).
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return {"metrics": metrics, "confusion_matrix": cm}
