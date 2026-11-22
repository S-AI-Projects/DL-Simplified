import json
import tempfile
import unittest
from pathlib import Path

from pipeline import BASE, load_config, load_fixture, run_pipeline


class TestReproduciblePipeline(unittest.TestCase):

    def test_fixture_has_exactly_six_records(self):
        self.assertEqual(len(load_fixture()), 6)

    def test_record_and_group_ids_are_unique(self):
        records = load_fixture()
        self.assertEqual(len({r["record_id"] for r in records}), 6)
        self.assertEqual(len({r["group_id"] for r in records}), 6)

    def test_fixture_has_three_records_per_class(self):
        labels = [r["label"] for r in load_fixture()]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_splits_are_disjoint_and_complete(self):
        config = load_config()
        splits = config["split_ids"]
        all_ids = sum(splits.values(), [])
        fixture_ids = {r["record_id"] for r in load_fixture()}
        self.assertEqual(len(all_ids), 6)
        self.assertEqual(set(all_ids), fixture_ids)

    def test_expected_predictions(self):
        results = run_pipeline()["results"]
        self.assertEqual(results["train"]["predictions"], [0, 0])
        self.assertEqual(results["validation"]["predictions"], [0, 1])
        self.assertEqual(results["test"]["predictions"], [1, 1])

    def test_reloaded_config_reproduces_results(self):
        first = run_pipeline()
        second = run_pipeline(BASE / "config.json")
        self.assertEqual(first, second)

    def test_unknown_top_level_field_is_rejected(self):
        config = json.loads((BASE / "config.json").read_text())
        config["unexpected"] = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_config(file.name)

    def test_unknown_algorithm_field_is_rejected(self):
        config = json.loads((BASE / "config.json").read_text())
        config["algorithm"]["extra"] = 5
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_config(file.name)

    def test_incompatible_image_shape_is_rejected(self):
        fixture = json.loads((BASE / "fixture.json").read_text())
        fixture["image_shape"] = [8, 8]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(fixture, file)
            file.flush()
            with self.assertRaises(ValueError):
                from pipeline import load_fixture
                load_fixture(file.name)

    def test_contradictory_split_ids_are_rejected(self):
        config = json.loads((BASE / "config.json").read_text())
        config["split_ids"]["test"][0] = "unknown-record"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            with self.assertRaises(ValueError):
                run_pipeline(file.name)

    def test_invalid_algorithm_is_rejected(self):
        config = json.loads((BASE / "config.json").read_text())
        config["algorithm"]["name"] = "unknown_algorithm"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(config, file)
            file.flush()
            with self.assertRaises(ValueError):
                load_config(file.name)


if __name__ == "__main__":
    unittest.main()
