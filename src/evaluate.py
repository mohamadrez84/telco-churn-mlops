from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def predict_labels(model, features, decision_threshold=None):
    if decision_threshold is None and not hasattr(model, "predict_proba"):
        predictions = model.predict(features)
        return predictions, predictions
    probabilities = model.predict_proba(features)[:, 1]
    if decision_threshold is None:
        predictions = model.predict(features)
    else:
        predictions = (probabilities >= decision_threshold).astype(int)
    return predictions, probabilities


def evaluate_model(model, X_test, y_test, decision_threshold=None) -> dict:
    y_pred, y_proba = predict_labels(model, X_test, decision_threshold)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return {"metrics": metrics, "confusion_matrix": cm}
