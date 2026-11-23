"""
Unit tests for R42-T008 — error-analysis report for the image fixture.

Offline, CPU-only, standard library only, deterministic (no randomness, so no
seed is needed), six-record fixture with a fixed train/validation/test split.

Run with:
    python3 -m unittest discover -s course-labs/r42-t008 -p "test_*.py"
"""

import json
import unittest

from pipeline import (
    LAB_DIR,
    check_aggregate,
    error_report,
    inject_wrong_prediction,
    load_fixture,
    load_json,
    make_predictions,
    run_report,
    split_records,
)


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


class TestReport(unittest.TestCase):
    def setUp(self):
        self.report = run_report(load_fixture(), load_json("split.json"))

    def test_model_alone_is_correct_on_held_out(self):
        rows = make_predictions(load_fixture(), load_json("split.json"))
        self.assertEqual([r["record_id"] for r in rows], ["r002", "r005", "r003", "r006"])
        self.assertTrue(all(r["predicted"] == r["expected"] for r in rows))
        self.assertEqual(error_report(rows)["rows"], [])

    def test_injected_wrong_prediction_appears_with_correct_record_id(self):
        self.assertEqual(self.report["rows"], [{
            "record_id": "r006",
            "split": "test",
            "expected": 1,
            "predicted": 0,
            "error_type": "false_negative",
            "reason_category": "injected_for_testing",
        }])

    def test_correct_predictions_do_not_appear(self):
        ids = {r["record_id"] for r in self.report["rows"]}
        for correct in ("r002", "r005", "r003"):
            self.assertNotIn(correct, ids)

    def test_training_records_are_not_evaluated(self):
        self.assertEqual(self.report["aggregate"]["evaluated"], 4)

    def test_aggregate_counts_match_rows(self):
        agg, rows = self.report["aggregate"], self.report["rows"]
        self.assertEqual(agg["errors"], len(rows))
        self.assertEqual(agg["correct"], 3)
        self.assertEqual(agg["by_split"], {"validation": 0, "test": 1})
        self.assertEqual(agg["by_error_type"], {"false_negative": 1})
        self.assertEqual(agg["by_reason_category"], {"injected_for_testing": 1})
        for key in ("by_split", "by_error_type", "by_reason_category"):
            self.assertEqual(sum(agg[key].values()), len(rows))

    def test_injecting_a_label0_record_gives_false_positive(self):
        report = run_report(load_fixture(), load_json("split.json"), record_id="r002")
        self.assertEqual([(r["record_id"], r["error_type"], r["split"]) for r in report["rows"]],
                         [("r002", "false_positive", "validation")])

    def test_saved_expected_report_matches(self):
        saved = json.loads((LAB_DIR / "expected_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.report)


class TestFailureCases(unittest.TestCase):
    def test_inject_unknown_record_rejected(self):
        rows = make_predictions(load_fixture(), load_json("split.json"))
        with self.assertRaisesRegex(ValueError, "'r999' is not an evaluated record"):
            inject_wrong_prediction(rows, "r999")

    def test_inject_training_record_rejected(self):
        rows = make_predictions(load_fixture(), load_json("split.json"))
        with self.assertRaisesRegex(ValueError, "'r001' is not an evaluated record"):
            inject_wrong_prediction(rows, "r001")

    def test_inject_twice_rejected(self):
        rows = inject_wrong_prediction(make_predictions(load_fixture(), load_json("split.json")), "r006")
        with self.assertRaisesRegex(ValueError, "already wrong"):
            inject_wrong_prediction(rows, "r006")

    def test_tampered_aggregate_detected(self):
        report = run_report(load_fixture(), load_json("split.json"))
        report["aggregate"]["by_split"]["validation"] = 1
        with self.assertRaisesRegex(ValueError, "by_split sums to 2, rows = 1"):
            check_aggregate(report["rows"], report["aggregate"])


if __name__ == "__main__":
    unittest.main()
