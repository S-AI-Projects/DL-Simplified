"""Unit tests for R42-T044 evaluation metric."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import compute_binary_metrics


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_canonical_metric_case(self):
        """Task specification requirement:
        truth [1, 1, 0, 0] / prediction [1, 0, 1, 0] -> precision=0.5, recall=0.5.
        """
        truth = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        res = compute_binary_metrics(truth, prediction)
        self.assertAlmostEqual(res["precision"], 0.5)
        self.assertAlmostEqual(res["recall"], 0.5)
        self.assertAlmostEqual(res["accuracy"], 0.5)
        self.assertAlmostEqual(res["f1"], 0.5)

    def test_zero_denominator_behavior(self):
        """When all predictions are 0 (no positive predictions), precision must handle zero division."""
        truth = [1, 1, 0, 0]
        prediction = [0, 0, 0, 0]
        res = compute_binary_metrics(truth, prediction, zero_division=0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f1"], 0.0)
        self.assertEqual(res["accuracy"], 0.5)

    def test_fixture_evaluation(self):
        """Verify metric evaluation on synthetic fixture labels."""
        y_true = [r["label"] for r in self.fixture]  # [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1]
        res = compute_binary_metrics(y_true, y_pred)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
