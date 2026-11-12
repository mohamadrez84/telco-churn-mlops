import unittest

from src.train import select_candidates


class ModelSelectionTest(unittest.TestCase):
    def test_accuracy_improvement_must_preserve_validation_f1(self):
        baseline = {"val_accuracy": 0.77, "val_f1": 0.65}
        balanced = {"val_accuracy": 0.82, "val_f1": 0.645}
        ignores_churn = {"val_accuracy": 0.87, "val_f1": 0.50}
        other = {"val_accuracy": 0.81, "val_f1": 0.66}

        selected, best, floor = select_candidates({
            ("v2", "A"): [baseline, balanced, ignores_churn],
            ("v3", "B"): [other, ignores_churn],
        })

        self.assertAlmostEqual(floor, 0.65)
        self.assertEqual(best, ("v3", "B"))
        self.assertEqual(selected[("v2", "A")], baseline)
        self.assertEqual(selected[("v3", "B")], other)
