"""Regression tests for the prediction request contract."""
import json
import unittest
from pathlib import Path

import pandas as pd

from deployment.app import app, bundle
from src.features import add_features
from src.preprocessing import encode_customers

CUSTOMER = json.loads((Path(__file__).resolve().parents[1] / "examples" / "customer.json").read_text())


class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_rejects_invalid_input(self):
        for payload in ({}, {"Tenure Months": 12}, {**CUSTOMER, "Contract": "Unknown"},
                        {**CUSTOMER, "Monthly Charges": -1}, {**CUSTOMER, "extra": 1}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 400)

    def test_raw_and_encoded_predictions_match(self):
        encoded = encode_customers(pd.DataFrame([CUSTOMER]))
        if bundle["dataset_version"] == "v3":
            encoded = add_features(encoded)
            threshold = bundle["high_value_threshold"]
            encoded["HighValueCustomer"] = (encoded["Monthly Charges"] > threshold).astype(int)
        payload = encoded.loc[0, bundle["feature_columns"]].to_dict()
        raw = self.client.post("/predict", json=CUSTOMER)
        processed = self.client.post("/predict", json=payload)
        self.assertEqual(raw.status_code, 200)
        self.assertEqual(processed.status_code, 200)
        self.assertEqual(raw.get_json(), processed.get_json())
