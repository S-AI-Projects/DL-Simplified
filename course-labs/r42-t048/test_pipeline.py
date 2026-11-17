"""Unit tests for R42-T048 error analysis report."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import generate_error_report


class TestErrorReporting(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_deliberate_injected_error_detection(self):
        """Inject exactly one deliberately wrong prediction (rec_002 expected 0, predicted 1).
        Verify that only rec_002 appears in the error report.
        """
        # Labels are [0, 0, 0, 1, 1, 1]
        injected_predictions = [0, 1, 0, 1, 1, 1]  # rec_002 is wrong
        report = generate_error_report(self.fixture, injected_predictions)

        self.assertEqual(report["total_records"], 6)
        self.assertEqual(report["correct_predictions"], 5)
        self.assertEqual(report["total_errors"], 1)
        self.assertAlmostEqual(report["error_rate"], 1/6)

        # Injected error must be rec_002
        self.assertEqual(len(report["error_records"]), 1)
        err = report["error_records"][0]
        self.assertEqual(err["record_id"], "rec_002")
        self.assertEqual(err["expected"], 0)
        self.assertEqual(err["predicted"], 1)
        self.assertEqual(err["reason_category"], "FALSE_POSITIVE_DIAGONAL_TRIGGER")

    def test_all_correct_predictions_empty_error_report(self):
        """When all predictions are correct, error report has 0 errors."""
        correct_predictions = [r["label"] for r in self.fixture]
        report = generate_error_report(self.fixture, correct_predictions)

        self.assertEqual(report["total_errors"], 0)
        self.assertEqual(report["error_records"], [])
        self.assertEqual(report["correct_predictions"], 6)


if __name__ == "__main__":
    unittest.main()
