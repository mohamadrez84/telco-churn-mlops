"""
app.py
------
فایل جدید — سرویس Inference با Flask (مرحله ۴ مستند پروژه: Model Deployment).
مدل ذخیره‌شده در models/best_model.pkl را می‌خواند و یک API ساده برای
پیش‌بینی ریزش مشتری فراهم می‌کند.

    POST /predict   -> پیش‌بینی برای یک مشتری (JSON)
    GET  /health     -> بررسی سلامت سرویس
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/best_model.pkl")
NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges", "AvgChargePerMonth"]

app = Flask(__name__)

print(f"[app] در حال بارگذاری مدل از {MODEL_PATH} ...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
print(f"[app] مدل بارگذاری شد. تعداد ویژگی: {len(feature_columns)}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "n_features": len(feature_columns)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        df = pd.DataFrame([payload])

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        cols_to_scale = [c for c in NUMERIC_COLUMNS if c in df.columns]
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0, 1]) if hasattr(model, "predict_proba") else None

        return jsonify({
            "churn_prediction": pred,
            "churn_probability": proba,
            "label": "Churn" if pred == 1 else "No Churn",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
