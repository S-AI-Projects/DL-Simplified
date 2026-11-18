import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import generate_error_report

class TestErrorReport(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_injected_error_identified(self):
        # Create predictions with exactly one deliberate error at aud_004 (label 1 predicted as 0)
        preds = {r["record_id"]: r["label"] for r in self.fixture}
        preds["aud_004"] = 0

        report = generate_error_report(self.fixture, preds)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(len(report["errors"]), 1)
        err = report["errors"][0]
        self.assertEqual(err["record_id"], "aud_004")
        self.assertEqual(err["expected"], 1)
        self.assertEqual(err["predicted"], 0)
        self.assertEqual(err["reason_category"], "false_negative")

    def test_perfect_predictions_zero_errors(self):
        preds = {r["record_id"]: r["label"] for r in self.fixture}
        report = generate_error_report(self.fixture, preds)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(len(report["errors"]), 0)

if __name__ == "__main__":
    unittest.main()
