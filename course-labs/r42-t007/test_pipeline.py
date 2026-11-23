"""
Unit tests for R42-T007 — reproducible configuration for the image fixture.

Offline, CPU-only, standard library only, fixed seed (42), six-record fixture,
explicit train/validation/test split recorded in config.json.

Run with:
    python3 -m unittest discover -s course-labs/r42-t007 -p "test_*.py"
"""

import copy
import json
import pathlib
import platform
import tempfile
import unittest

from pipeline import (
    LAB_DIR,
    load_config,
    load_fixture,
    make_config,
    run_from_config,
    save_config,
    validate_config,
)

EXPECTED_SPLIT = {"train": ["r002", "r005"], "validation": ["r001", "r004"], "test": ["r003", "r006"]}
EXPECTED_PREDICTIONS = {"r001": 0, "r004": 1, "r003": 0, "r006": 1}


class TestFixture(unittest.TestCase):
    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)
        self.assertEqual({r["group_id"] for r in fixture}, {"g1", "g2", "g3"})
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], [0] * 16 if rec["label"] == 0 else diagonal)


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        self.fixture = load_fixture()
        self.config = load_config(LAB_DIR / "config.json")

    def test_config_records_required_settings(self):
        c = self.config
        self.assertEqual(c["schema_version"], "1.0")
        self.assertEqual(c["seed"], 42)
        self.assertEqual(c["input_shape"], [4, 4])
        self.assertEqual(c["feature_order"][:5], ["p00", "p01", "p02", "p03", "p10"])
        self.assertEqual(len(c["feature_order"]), 16)
        self.assertEqual(c["split"], EXPECTED_SPLIT)
        self.assertEqual(c["algorithm"]["name"], "nearest_centroid")

    def test_environment_versions_recorded(self):
        self.assertEqual(set(self.config["environment"]), {"python", "platform"})
        fresh = make_config(self.fixture)
        self.assertEqual(fresh["environment"]["python"], platform.python_version())

    def test_saved_config_reconstructs_split_and_predictions(self):
        result = run_from_config(self.config, self.fixture)
        self.assertEqual(result["split"], EXPECTED_SPLIT)
        self.assertEqual(result["predictions"], EXPECTED_PREDICTIONS)

    def test_save_reload_round_trip_gives_same_result(self):
        fresh = make_config(self.fixture, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.json"
            save_config(fresh, path)
            reloaded = load_config(path)
        self.assertEqual(reloaded, fresh)
        self.assertEqual(run_from_config(reloaded, self.fixture), run_from_config(fresh, self.fixture))

    def test_regenerated_config_matches_committed_config(self):
        fresh = make_config(self.fixture, seed=42)
        committed = copy.deepcopy(self.config)
        del fresh["environment"], committed["environment"]   # machine-specific, recorded only
        self.assertEqual(fresh, committed)

    def test_saved_expected_result_matches(self):
        saved = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, run_from_config(self.config, self.fixture))


class TestRejectedSettings(unittest.TestCase):
    def setUp(self):
        self.config = load_config(LAB_DIR / "config.json")
        self.fixture = load_fixture()

    def test_unknown_top_level_field_rejected(self):
        self.config["learning_rate"] = 0.1
        with self.assertRaisesRegex(ValueError, r"unknown config field\(s\): \['learning_rate'\]"):
            validate_config(self.config)

    def test_unknown_algorithm_field_rejected(self):
        self.config["algorithm"]["k"] = 3
        with self.assertRaisesRegex(ValueError, r"unknown algorithm field\(s\): \['k'\]"):
            validate_config(self.config)

    def test_missing_field_rejected(self):
        del self.config["seed"]
        with self.assertRaisesRegex(ValueError, "missing config field"):
            validate_config(self.config)

    def test_unsupported_schema_version_rejected(self):
        self.config["schema_version"] = "2.0"
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            validate_config(self.config)

    def test_feature_order_contradicting_shape_rejected(self):
        self.config["input_shape"] = [2, 8]
        with self.assertRaisesRegex(ValueError, "feature_order does not match"):
            validate_config(self.config)

    def test_overlapping_split_ids_rejected(self):
        self.config["split"]["test"].append("r002")
        with self.assertRaisesRegex(ValueError, "'r002' is in both 'train' and 'test'"):
            validate_config(self.config)

    def test_non_positive_pixel_scale_rejected(self):
        self.config["algorithm"]["pixel_scale"] = 0
        with self.assertRaisesRegex(ValueError, "pixel_scale must be > 0"):
            validate_config(self.config)

    def test_incompatible_input_shape_rejected(self):
        records = copy.deepcopy(self.fixture)
        records[0]["pixels"] = [0] * 9                        # a 3×3 image
        with self.assertRaisesRegex(ValueError, "incompatible input shape"):
            run_from_config(self.config, records)

    def test_nested_4x4_input_rejected(self):
        records = copy.deepcopy(self.fixture)
        records[2]["pixels"] = [[0] * 4 for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "incompatible input shape"):
            run_from_config(self.config, records)

    def test_record_not_in_configured_split_rejected(self):
        records = copy.deepcopy(self.fixture)
        records.append(dict(records[0], record_id="r999"))
        with self.assertRaisesRegex(ValueError, r"unassigned=\['r999'\]"):
            run_from_config(self.config, records)


if __name__ == "__main__":
    unittest.main()
