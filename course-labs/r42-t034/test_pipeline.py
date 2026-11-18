import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import compute_classification_metrics, compute_mae

class TestAudioMetrics(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_canonical_classification_example(self):
        truth = [1, 1, 0, 0]
        pred = [1, 0, 1, 0]
        metrics = compute_classification_metrics(truth, pred)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)

    def test_canonical_mae_example(self):
        truth = [1, 2, 3]
        pred = [1, 3, 2]
        mae = compute_mae(truth, pred)
        self.assertAlmostEqual(mae, 2.0 / 3.0)

    def test_zero_denominator_safe(self):
        truth = [0, 0, 0]
        pred = [0, 0, 0]
        metrics = compute_classification_metrics(truth, pred)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_audio_fixture_schema(self):
        self.assertEqual(len(self.fixture), 6)
        for r in self.fixture:
            self.assertEqual(r["sample_rate"], 8000)
            self.assertEqual(len(r["samples"]), 8)

if __name__ == "__main__":
    unittest.main()
