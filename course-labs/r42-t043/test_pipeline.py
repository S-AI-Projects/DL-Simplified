"""Unit tests for R42-T043 minimal baseline."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import MajorityClassBaseline, evaluate_predictions


class TestMajorityBaseline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_baseline_parameters_trained_only_on_train_split(self):
        """Train split contains 3 records with label 0 and 1 record with label 1.
        Baseline must learn majority class = 0 strictly from train records.
        """
        train_records = self.fixture[:4]  # rec_001, 002, 003 (label 0), rec_004 (label 1)
        test_records = self.fixture[4:]   # rec_005, 006 (label 1)

        model = MajorityClassBaseline()
        model.fit(train_records)
        self.assertEqual(model.majority_label, 0)

        preds = model.predict(test_records)
        self.assertEqual(preds, [0, 0])

        metrics = evaluate_predictions([r["label"] for r in test_records], preds)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_balanced_or_unseen_heldout(self):
        """Verify baseline prediction on held-out test data."""
        # Train on first 3 (all label 0)
        train_data = self.fixture[:3]
        test_data = self.fixture[3:]
        model = MajorityClassBaseline().fit(train_data)
        self.assertEqual(model.majority_label, 0)

        preds = model.predict(test_data)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(p == 0 for p in preds))

    def test_evaluate_predictions_perfect(self):
        """Verify evaluation metrics on perfect match."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
