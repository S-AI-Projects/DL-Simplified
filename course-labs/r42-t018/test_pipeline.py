import json
import tempfile
import unittest
from pathlib import Path

from pipeline import BASE, load_fixture, run_pipeline


class TestErrorAnalysis(unittest.TestCase):

    def test_fixture_has_six_records(self):
        self.assertEqual(len(load_fixture()), 6)

    def test_ids_are_unique(self):
        records = load_fixture()
        self.assertEqual(len({r["record_id"] for r in records}), 6)
        self.assertEqual(len({r["group_id"] for r in records}), 6)

    def test_three_records_per_class(self):
        labels = [r["label"] for r in load_fixture()]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_exactly_one_error(self):
        report = run_pipeline()
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["correct_count"], 5)

    def test_injected_error_details(self):
        report = run_pipeline()
        self.assertEqual(report["errors"], [{
            "record_id": "diagonal-003",
            "expected": 1,
            "predicted": 0,
            "reason_category": "injected_prediction_error"
        }])

    def test_correct_predictions_do_not_appear(self):
        report = run_pipeline()
        error_ids = {row["record_id"] for row in report["errors"]}
        self.assertNotIn("zero-001", error_ids)
        self.assertNotIn("diagonal-001", error_ids)

    def test_error_counts_match_report_rows(self):
        report = run_pipeline()
        self.assertEqual(report["error_count"], len(report["errors"]))
        self.assertEqual(
            report["correct_count"] + report["error_count"],
            report["total_records"]
        )

    def test_fixed_seed_is_reproducible(self):
        self.assertEqual(run_pipeline(), run_pipeline())

    def test_wrong_image_shape_is_rejected(self):
        fixture = json.loads((BASE / "fixture.json").read_text())
        fixture["image_shape"] = [8, 8]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(fixture, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_fixture(file.name)

    def test_duplicate_record_ids_are_rejected(self):
        fixture = json.loads((BASE / "fixture.json").read_text())
        fixture["records"][1]["record_id"] = fixture["records"][0]["record_id"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(fixture, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_fixture(file.name)

    def test_invalid_label_is_rejected(self):
        fixture = json.loads((BASE / "fixture.json").read_text())
        fixture["records"][0]["label"] = 9
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(fixture, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_fixture(file.name)


if __name__ == "__main__":
    unittest.main()
