"""Serve churn predictions using the transformations fitted during training."""
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.features import prepare_raw_features, transform_features
from src.preprocessing import RAW_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "best_model.pkl"))

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]


@app.get("/health")
def health():
    return jsonify({"status": "ok", "n_features": len(feature_columns)})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Expected a JSON object"), 400

    try:
        if set(payload) == set(feature_columns):
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in payload.values()):
                raise ValueError("Encoded features must be finite numbers")
            df = transform_features(pd.DataFrame([payload]), scaler, None)
        elif RAW_COLUMNS <= payload.keys():
            if payload.keys() - RAW_COLUMNS:
                raise ValueError(f"Unexpected fields: {', '.join(sorted(payload.keys() - RAW_COLUMNS))}")
            df = prepare_raw_features(pd.DataFrame([payload]), bundle)
        else:
            raise ValueError(f"Missing required fields: {', '.join(sorted(RAW_COLUMNS - payload.keys()))}")
    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400

    df = df[feature_columns]
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])
    return jsonify(
        churn_prediction=prediction,
        churn_probability=probability,
        label="Churn" if prediction == 1 else "No Churn",
    )


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8000)
