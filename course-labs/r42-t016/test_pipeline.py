import json
import tempfile
import unittest
from pathlib import Path

from pipeline import load_fixture, run_smoke


class TestSmokePipeline(unittest.TestCase):

    def test_fixture_has_six_records_and_three_splits(self):
        data = load_fixture()
        self.assertEqual(len(data["records"]), 6)
        self.assertEqual(
            {r["split"] for r in data["records"]},
            {"train", "validation", "test"},
        )

    def test_fixture_has_expected_labels(self):
        data = load_fixture()
        labels = [r["label"] for r in data["records"]]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_pipeline_output_is_finite_and_bounded(self):
        result = run_smoke()
        self.assertEqual(result["sample_count"], 6)
        self.assertEqual(result["prediction_shape"], [6])
        self.assertEqual(result["step_count"], 0)
        self.assertLessEqual(result["sample_count"], 32)
        self.assertLessEqual(result["step_count"], 20)
        self.assertTrue(0 <= result["loss"] < float("inf"))

    def test_predictions_match_synthetic_fixture(self):
        result = run_smoke()
        self.assertEqual(result["predictions"], [0, 0, 1, 0, 1, 1])

    def test_fixed_seed_is_reproducible(self):
        self.assertEqual(run_smoke(seed=42), run_smoke(seed=42))

    def test_duplicate_record_ids_are_rejected(self):
        data = load_fixture()
        data["records"][1]["record_id"] = data["records"][0]["record_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
                load_fixture(path)

    def test_duplicate_group_ids_are_rejected(self):
        data = load_fixture()
        data["records"][1]["group_id"] = data["records"][0]["group_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate group IDs"):
                load_fixture(path)

    def test_invalid_pixel_range_is_rejected(self):
        data = load_fixture()
        data["records"][0]["pixels"][0][0] = 256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "range"):
                load_fixture(path)

    def test_invalid_image_shape_is_rejected(self):
        data = load_fixture()
        data["records"][0]["pixels"] = [[0, 0], [0, 0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "4x4"):
                load_fixture(path)

    def test_invalid_label_is_rejected(self):
        data = load_fixture()
        data["records"][0]["label"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label"):
                load_fixture(path)


if __name__ == "__main__":
    unittest.main()
