import unittest

import pandas as pd

from deployment.app import bundle
from evaluate_external import evaluate_external
from test_prediction import CUSTOMER


class ExternalEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.reference = pd.Series(["IBM-001"])
        self.cohort = pd.DataFrame([
            {**CUSTOMER, "CustomerID": "EX-01", "Churn Value": 0},
            {**CUSTOMER, "CustomerID": "EX-02", "Churn Value": 1},
        ])

    def test_reports_only_aggregate_results(self):
        result = evaluate_external(self.cohort, self.reference, bundle)
        self.assertEqual(result["n_customers"], 2)
        self.assertEqual(result["n_churn"], 1)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 2)

    def test_rejects_overlap_even_with_case_and_whitespace(self):
        self.cohort.loc[0, "CustomerID"] = " ibm-001 "
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_external(self.cohort, self.reference, bundle)

    def test_rejects_a_single_class(self):
        self.cohort["Churn Value"] = 0
        with self.assertRaisesRegex(ValueError, "both classes"):
            evaluate_external(self.cohort, self.reference, bundle)
