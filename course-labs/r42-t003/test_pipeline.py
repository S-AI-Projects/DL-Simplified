"""
Unit tests for R42-T003 — majority-class baseline for the image contract.

Offline, CPU-only, standard library only, deterministic (no randomness, so no
seed is needed), six-record fixture with a fixed train/validation/test split.

Run with:
    python3 -m unittest discover -s course-labs/r42-t003 -p "test_*.py"
"""

import copy
import json
import unittest

from pipeline import (
    LAB_DIR,
    MajorityClassBaseline,
    accuracy,
    load_fixture,
    load_split,
    run_baseline,
    split_records,
)


class TestFixture(unittest.TestCase):
    def test_six_records_three_per_label_with_expected_pixels(self):
        fixture = load_fixture()
        zero = [0] * 16
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], zero if rec["label"] == 0 else diagonal)
        self.assertEqual(sorted(r["label"] for r in fixture), [0, 0, 0, 1, 1, 1])
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)


class TestSplit(unittest.TestCase):
    def test_explicit_disjoint_split(self):
        parts = split_records(load_fixture(), load_split())
        ids = {k: sorted(r["record_id"] for r in v) for k, v in parts.items()}
        self.assertEqual(ids, {
            "train": ["r001", "r002", "r004"],
            "validation": ["r005"],
            "test": ["r003", "r006"],
        })

    def test_group_in_two_splits_rejected(self):
        split = load_split()
        split["test"].append("g1")
        with self.assertRaisesRegex(ValueError, "group_id 'g1' is in both 'train' and 'test'"):
            split_records(load_fixture(), split)

    def test_duplicate_record_id_rejected(self):
        records = load_fixture()
        records.append(copy.deepcopy(records[0]))
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            split_records(records, load_split())


class TestBaselineResult(unittest.TestCase):
    def setUp(self):
        self.result = run_baseline(load_fixture(), load_split())

    def test_fitted_on_training_records_only(self):
        self.assertEqual(self.result["fitted_on"], ["r001", "r002", "r004"])
        self.assertEqual(self.result["train_label_counts"], {"0": 2, "1": 1})
        self.assertEqual(self.result["majority_label"], 0)

    def test_validation_predictions_and_accuracy(self):
        self.assertEqual(self.result["validation"]["truth"], [1])
        self.assertEqual(self.result["validation"]["prediction"], [0])
        self.assertEqual(self.result["validation"]["accuracy"], 0.0)

    def test_test_predictions_and_accuracy(self):
        self.assertEqual(self.result["test"]["record_ids"], ["r003", "r006"])
        self.assertEqual(self.result["test"]["truth"], [0, 1])
        self.assertEqual(self.result["test"]["prediction"], [0, 0])
        self.assertEqual(self.result["test"]["accuracy"], 0.5)

    def test_held_out_labels_do_not_change_fitted_parameter(self):
        records = load_fixture()
        for rec in records:
            if rec["group_id"] in ("g2", "g3"):      # validation + test
                rec["label"] = 1
        changed = run_baseline(records, load_split())
        self.assertEqual(changed["majority_label"], 0)
        self.assertEqual(changed["train_label_counts"], {"0": 2, "1": 1})

    def test_saved_expected_result_matches(self):
        saved = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.result)


class TestFailureCases(unittest.TestCase):
    def test_tie_breaks_to_smallest_label(self):
        model = MajorityClassBaseline().fit([{"label": 1}, {"label": 0}])
        self.assertEqual(model.majority_label, 0)

    def test_empty_training_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty training split"):
            MajorityClassBaseline().fit([])

    def test_invalid_training_label_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid training label"):
            MajorityClassBaseline().fit([{"label": 2}])

    def test_predict_before_fit_rejected(self):
        with self.assertRaises(RuntimeError):
            MajorityClassBaseline().predict([{}])

    def test_accuracy_rejects_empty_and_mismatched(self):
        with self.assertRaisesRegex(ValueError, "zero records"):
            accuracy([], [])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            accuracy([0, 1], [0])


if __name__ == "__main__":
    unittest.main()
