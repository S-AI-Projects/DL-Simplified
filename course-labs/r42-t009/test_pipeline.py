"""
Unit tests for R42-T009 — record-order robustness and shape-mismatch rejection.

Offline, CPU-only, standard library only, fixed permutation seed (7),
six-record fixture with a fixed train/validation/test split.

Run with:
    python3 -m unittest discover -s course-labs/r42-t009 -p "test_*.py"
"""

import copy
import json
import unittest

from pipeline import (
    LAB_DIR,
    check_input,
    load_fixture,
    load_json,
    permute,
    predict_by_id,
)

EXPECTED = {"r002": 0, "r003": 0, "r005": 1, "r006": 1}


class TestFixture(unittest.TestCase):
    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], [0] * 16 if rec["label"] == 0 else diagonal)


class TestRecordOrderProperty(unittest.TestCase):
    """Valid transformation: reordering records with stable IDs."""

    def setUp(self):
        self.fixture = load_fixture()
        self.split = load_json("split.json")

    def test_documented_result(self):
        self.assertEqual(predict_by_id(self.fixture, self.split), EXPECTED)

    def test_seeded_shuffle_preserves_result_after_id_alignment(self):
        shuffled = permute(self.fixture, seed=7)
        self.assertEqual([r["record_id"] for r in shuffled], ["r005", "r001", "r006", "r004", "r002", "r003"])
        result = predict_by_id(shuffled, self.split)
        self.assertNotEqual(list(result), list(EXPECTED))      # raw output order differs ...
        self.assertEqual(result, EXPECTED)                     # ... but aligned by ID it is identical

    def test_reversed_order_preserves_result(self):
        self.assertEqual(predict_by_id(list(reversed(self.fixture)), self.split), EXPECTED)

    def test_ids_stay_attached_to_records(self):
        by_id = {r["record_id"]: r for r in self.fixture}
        for rec in permute(self.fixture):
            self.assertIs(rec, by_id[rec["record_id"]])

    def test_saved_expected_result_matches(self):
        saved = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["property_holds"])
        self.assertEqual(saved["original_predictions"], EXPECTED)
        self.assertEqual(saved["shuffled_predictions_aligned"], EXPECTED)


class TestUnsupportedInput(unittest.TestCase):
    """Unsupported input: shape mismatch is rejected, never padded or reshaped."""

    def setUp(self):
        self.fixture = copy.deepcopy(load_fixture())
        self.split = load_json("split.json")

    def test_fifteen_pixels_rejected(self):
        self.fixture[5]["pixels"] = self.fixture[5]["pixels"][:15]
        with self.assertRaisesRegex(ValueError, "'r006': unsupported input, shape mismatch: .* got 15"):
            predict_by_id(self.fixture, self.split)

    def test_nested_4x4_rejected(self):
        self.fixture[2]["pixels"] = [[0] * 4 for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            predict_by_id(self.fixture, self.split)

    def test_missing_pixels_field_rejected(self):
        del self.fixture[1]["pixels"]
        with self.assertRaisesRegex(ValueError, "'r002': unsupported input, missing pixels field"):
            predict_by_id(self.fixture, self.split)

    def test_valid_input_accepted(self):
        self.assertEqual(len(check_input(self.fixture[0])), 16)


if __name__ == "__main__":
    unittest.main()
