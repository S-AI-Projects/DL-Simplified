import json
import tempfile
import unittest
from pathlib import Path

from pipeline import load_fixture, predict, run_pipeline


class TestInputPerturbation(unittest.TestCase):

    def test_fixture_contains_six_records(self):
        self.assertEqual(len(load_fixture()), 6)

    def test_record_ids_are_unique(self):
        records = load_fixture()
        ids = [r["record_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_group_ids_are_unique(self):
        records = load_fixture()
        ids = [r["group_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_fixture_has_three_records_per_label(self):
        records = load_fixture()
        labels = [r["label"] for r in records]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_predictions_match_expected_labels(self):
        for record in load_fixture():
            self.assertEqual(predict(record), record["label"])

    def test_order_perturbation_preserves_predictions(self):
        result = run_pipeline()
        self.assertTrue(result["order_changed"])
        self.assertTrue(result["stable_after_alignment"])
        self.assertEqual(
            result["baseline_predictions"],
            result["shuffled_predictions"]
        )

    def test_seeded_shuffle_is_reproducible(self):
        first = run_pipeline()
        second = run_pipeline()
        self.assertEqual(
            first["shuffled_predictions"],
            second["shuffled_predictions"]
        )

    def test_missing_pixels_field_is_rejected(self):
        records = load_fixture()
        del records[0]["pixels"]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({
                "image_shape": [4, 4],
                "records": records
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_fixture(path)

    def test_wrong_image_shape_is_rejected(self):
        fixture = {
            "image_shape": [3, 3],
            "records": []
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_fixture(path)

    def test_invalid_pixel_value_is_rejected(self):
        records = load_fixture()
        records[0]["pixels"][0][0] = 300

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({
                "image_shape": [4, 4],
                "records": records
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_fixture(path)


if __name__ == "__main__":
    unittest.main()
