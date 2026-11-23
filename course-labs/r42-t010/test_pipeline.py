"""
Unit tests for R42-T010 — saved-pipeline interface (JSON only, no pickle).

Offline, CPU-only, standard library only, deterministic (no randomness, so no
seed is needed), six-record fixture with a fixed train/validation/test split.

Run with:
    python3 -m unittest discover -s course-labs/r42-t010 -p "test_*.py"
"""

import copy
import json
import pathlib
import tempfile
import unittest

from pipeline import (
    LAB_DIR,
    TOLERANCE,
    fit_pipeline,
    inference_inputs,
    load_fixture,
    load_json,
    load_pipeline,
    predict,
    predictions_agree,
    save_pipeline,
    split_records,
)


def held_out_inputs():
    parts = split_records(load_fixture(), load_json("split.json"))
    return inference_inputs(parts["validation"] + parts["test"])


class TestFixture(unittest.TestCase):
    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], [0] * 16 if rec["label"] == 0 else diagonal)

    def test_explicit_disjoint_split(self):
        parts = split_records(load_fixture(), load_json("split.json"))
        self.assertEqual({k: [r["record_id"] for r in v] for k, v in parts.items()}, {
            "train": ["r001", "r004"], "validation": ["r002", "r005"], "test": ["r003", "r006"]})


class TestSaveReload(unittest.TestCase):
    def setUp(self):
        parts = split_records(load_fixture(), load_json("split.json"))
        self.pipe = fit_pipeline(parts["train"])
        self.inputs = held_out_inputs()

    def test_committed_saved_pipeline_matches_fit(self):
        self.assertEqual(load_pipeline(LAB_DIR / "saved_pipeline.json"), self.pipe)

    def test_before_and_after_reload_agree_within_tolerance(self):
        before = predict(self.pipe, self.inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pipe.json"
            save_pipeline(self.pipe, path)
            after = predict(load_pipeline(path), self.inputs)
        self.assertEqual(TOLERANCE, 1e-9)
        self.assertTrue(predictions_agree(before, after))

    def test_concrete_predictions(self):
        preds = predict(load_pipeline(LAB_DIR / "saved_pipeline.json"), self.inputs)
        self.assertEqual([(p["record_id"], p["label"]) for p in preds],
                         [("r002", 0), ("r005", 1), ("r003", 0), ("r006", 1)])
        self.assertEqual(preds[1]["scores"], {"0": 4.0, "1": 0.0})

    def test_tolerance_is_enforced(self):
        before = predict(self.pipe, self.inputs)
        tiny, large = copy.deepcopy(self.pipe), copy.deepcopy(self.pipe)
        tiny["params"]["centroids"]["1"][0] += 1e-12          # well inside 1e-9
        large["params"]["centroids"]["1"][0] += 1e-3          # far outside 1e-9
        self.assertTrue(predictions_agree(before, predict(tiny, self.inputs)))
        self.assertFalse(predictions_agree(before, predict(large, self.inputs)))

    def test_inference_needs_no_training_data_or_labels(self):
        # Only the saved file is in the directory; inputs carry no label or group_id.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pipe.json"
            path.write_text((LAB_DIR / "saved_pipeline.json").read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual([p.name for p in pathlib.Path(tmp).iterdir()], ["pipe.json"])
            preds = predict(load_pipeline(path), self.inputs)
        self.assertTrue(all(set(i) == {"record_id", "pixels"} for i in self.inputs))
        self.assertEqual(len(preds), 4)
        self.assertEqual(set(self.pipe), {"format", "format_version", "schema", "params"})

    def test_saved_expected_result_matches(self):
        saved = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["agree_within_tolerance"])
        self.assertEqual(saved["after_reload"], predict(self.pipe, self.inputs))


class TestClearFailures(unittest.TestCase):
    def setUp(self):
        self.pipe = load_pipeline(LAB_DIR / "saved_pipeline.json")

    def _load_modified(self, change):
        bad = copy.deepcopy(self.pipe)
        change(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            return load_pipeline(path)

    def test_version_mismatch_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "version mismatch: file has format_version 2, this code supports 1"):
            self._load_modified(lambda p: p.update(format_version=2))

    def test_schema_mismatch_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self._load_modified(lambda p: p["schema"].update(pixels_length=9))

    def test_wrong_format_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "format mismatch"):
            self._load_modified(lambda p: p.update(format="something-else"))

    def test_pickle_file_refused_without_opening(self):
        with self.assertRaisesRegex(ValueError, "only .json pipelines are supported"):
            load_pipeline(pathlib.Path("does_not_exist.pkl"))

    def test_input_with_wrong_pixel_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "input schema: record 'x1' needs 16 pixels, got 15"):
            predict(self.pipe, [{"record_id": "x1", "pixels": [0] * 15}])

    def test_input_missing_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "input schema: missing field 'pixels'"):
            predict(self.pipe, [{"record_id": "x1"}])

    def test_input_out_of_range_pixel_rejected(self):
        with self.assertRaisesRegex(ValueError, "pixel 256 is not an int in \\[0, 255\\]"):
            predict(self.pipe, [{"record_id": "x1", "pixels": [256] + [0] * 15}])


if __name__ == "__main__":
    unittest.main()
