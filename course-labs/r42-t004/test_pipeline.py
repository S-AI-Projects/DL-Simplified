"""
Unit tests for R42-T004 — metric correctness for the image fixture.

Offline, CPU-only, standard library only, deterministic (no randomness, so no
seed is needed). Includes an exhaustive cross-check against an independent
reference implementation written separately below.

Run with:
    python3 -m unittest discover -s course-labs/r42-t004 -p "test_*.py"
"""

import itertools
import json
import unittest

from pipeline import (
    LAB_DIR,
    accuracy,
    counts,
    evaluate_fixture,
    f1,
    load_fixture,
    load_predictions,
    precision,
    recall,
)


def reference(truth, pred, cls, zero_division=0.0):
    """Independent set-based reference: returns (precision, recall, f1) for class cls."""
    actual = {i for i, t in enumerate(truth) if t == cls}
    predicted = {i for i, p in enumerate(pred) if p == cls}
    hit = len(actual & predicted)
    p = hit / len(predicted) if predicted else zero_division
    r = hit / len(actual) if actual else zero_division
    denom = len(actual) + len(predicted)            # = 2TP + FP + FN
    f = 2 * hit / denom if denom else zero_division
    return p, r, f


class TestTaskExample(unittest.TestCase):
    """truth [1,1,0,0] / prediction [1,0,1,0] → precision = recall = 0.5."""

    def test_precision_is_half(self):
        self.assertEqual(precision([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)

    def test_recall_is_half(self):
        self.assertEqual(recall([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)

    def test_counts(self):
        self.assertEqual(counts([1, 1, 0, 0], [1, 0, 1, 0]), (1, 1, 1))

    def test_macro_and_f1_and_accuracy(self):
        self.assertEqual(precision([1, 1, 0, 0], [1, 0, 1, 0], average="macro"), 0.5)
        self.assertEqual(f1([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)
        self.assertEqual(accuracy([1, 1, 0, 0], [1, 0, 1, 0]), 0.5)


class TestAveraging(unittest.TestCase):
    """Binary and macro differ when the classes behave differently."""

    def test_binary_vs_macro(self):
        truth, pred = [1, 1, 1, 0], [1, 1, 1, 1]
        # class 1: P = 3/4, R = 1;  class 0: no predictions → P = 0 (zero_division), R = 0
        self.assertEqual(precision(truth, pred, average="binary"), 0.75)
        self.assertEqual(precision(truth, pred, average="macro"), (0.75 + 0.0) / 2)
        self.assertEqual(recall(truth, pred, average="macro"), (1.0 + 0.0) / 2)

    def test_unknown_average_rejected(self):
        with self.assertRaisesRegex(ValueError, "average must be"):
            precision([1], [1], average="weighted")


class TestZeroDenominator(unittest.TestCase):
    """No positives at all: every denominator for class 1 is 0."""

    truth, pred = [0, 0], [0, 0]

    def test_default_returns_zero(self):
        self.assertEqual(precision(self.truth, self.pred), 0.0)
        self.assertEqual(recall(self.truth, self.pred), 0.0)
        self.assertEqual(f1(self.truth, self.pred), 0.0)

    def test_zero_division_one(self):
        self.assertEqual(precision(self.truth, self.pred, zero_division=1.0), 1.0)
        # macro: class 0 is perfect (1.0) and class 1 uses zero_division
        self.assertEqual(recall(self.truth, self.pred, average="macro", zero_division=0.0), 0.5)
        self.assertEqual(recall(self.truth, self.pred, average="macro", zero_division=1.0), 1.0)

    def test_zero_division_raise(self):
        with self.assertRaises(ZeroDivisionError):
            precision(self.truth, self.pred, zero_division="raise")

    def test_invalid_zero_division_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero_division"):
            precision(self.truth, self.pred, zero_division=0.5)


class TestInputValidation(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero records"):
            accuracy([], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            precision([1, 0], [1])

    def test_non_label_values_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a label"):
            recall([1, 2], [1, 0])
        with self.assertRaisesRegex(ValueError, "not a label"):
            recall([True, 0], [1, 0])


class TestCrossCheck(unittest.TestCase):
    """Exhaustive check of all 256 length-4 truth/prediction pairs vs the reference."""

    def test_matches_reference_for_all_pairs(self):
        vectors = [list(v) for v in itertools.product((0, 1), repeat=4)]
        for truth in vectors:
            for pred in vectors:
                for zd in (0.0, 1.0):
                    for cls_avg in ("binary", "macro"):
                        if cls_avg == "binary":
                            exp = reference(truth, pred, 1, zd)
                        else:
                            r0, r1 = reference(truth, pred, 0, zd), reference(truth, pred, 1, zd)
                            exp = tuple((a + b) / 2 for a, b in zip(r0, r1))
                        got = (
                            precision(truth, pred, cls_avg, zd),
                            recall(truth, pred, cls_avg, zd),
                            f1(truth, pred, cls_avg, zd),
                        )
                        for g, e in zip(got, exp):
                            self.assertAlmostEqual(g, e, places=12, msg=(truth, pred, zd, cls_avg))


class TestFixtureEvaluation(unittest.TestCase):
    def setUp(self):
        self.result = evaluate_fixture(load_fixture(), load_predictions())

    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], [0] * 16 if rec["label"] == 0 else diagonal)

    def test_concrete_fixture_metrics(self):
        self.assertEqual(self.result["truth"], [0, 0, 0, 1, 1, 1])
        self.assertEqual(self.result["prediction"], [0, 1, 0, 1, 0, 1])
        self.assertEqual(self.result["positive_class_counts"], {"tp": 2, "fp": 1, "fn": 1})
        m = self.result["metrics"]
        for key in ("accuracy", "precision_binary", "recall_binary", "f1_binary",
                    "precision_macro", "recall_macro", "f1_macro"):
            self.assertAlmostEqual(m[key], 2 / 3, places=12, msg=key)

    def test_missing_prediction_rejected(self):
        preds = load_predictions()
        del preds["r006"]
        with self.assertRaisesRegex(ValueError, r"missing=\['r006'\]"):
            evaluate_fixture(load_fixture(), preds)

    def test_saved_expected_result_matches(self):
        saved = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["fixture"], self.result)
        self.assertEqual(saved["task_example"]["precision_binary"], 0.5)
        self.assertEqual(saved["task_example"]["recall_binary"], 0.5)


if __name__ == "__main__":
    unittest.main()
