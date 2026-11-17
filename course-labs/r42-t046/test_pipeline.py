"""Unit tests for R42-T046 bounded CPU smoke run."""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import TinyLinearClassifier, flatten_record_pixels


class TestCPUSmokeRun(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_bounded_training_execution(self):
        """Execute bounded CPU smoke run: <= 20 update steps, 1 epoch, finite loss."""
        features, labels = flatten_record_pixels(self.fixture)
        self.assertEqual(len(features), 6)
        self.assertEqual(len(features[0]), 16)

        model = TinyLinearClassifier(input_dim=16, seed=42)
        summary = model.train_bounded(features, labels, max_steps=10)

        # Recorded steps must be bounded (<= 20 steps, exactly 6 for 6 records)
        self.assertLessEqual(summary["steps_recorded"], 20)
        self.assertEqual(summary["steps_recorded"], 6)
        self.assertTrue(summary["loss_is_finite"])
        self.assertTrue(math.isfinite(summary["average_loss"]))

    def test_prediction_shape_and_range(self):
        """Predictions must match input count and consist of valid binary labels."""
        features, _ = flatten_record_pixels(self.fixture)
        model = TinyLinearClassifier(input_dim=16, seed=42)
        preds = model.predict(features)

        self.assertEqual(len(preds), 6)
        self.assertTrue(all(p in (0, 1) for p in preds))

    def test_deterministic_reproducibility(self):
        """Fixed seed must produce exact same outputs across runs."""
        features, labels = flatten_record_pixels(self.fixture)
        m1 = TinyLinearClassifier(input_dim=16, seed=42)
        s1 = m1.train_bounded(features, labels)
        p1 = m1.predict(features)

        m2 = TinyLinearClassifier(input_dim=16, seed=42)
        s2 = m2.train_bounded(features, labels)
        p2 = m2.predict(features)

        self.assertEqual(s1["average_loss"], s2["average_loss"])
        self.assertEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
