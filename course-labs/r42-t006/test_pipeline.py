"""
Unit tests for R42-T006 — bounded CPU smoke run on the image fixture.

Offline, CPU-only, standard library only, fixed seed (0), six records,
one epoch, two update steps. Asserts interface correctness, not accuracy.

Run with:
    python3 -m unittest discover -s course-labs/r42-t006 -p "test_*.py"
"""

import copy
import math
import unittest

from pipeline import (
    MAX_SAMPLES,
    MAX_STEPS,
    load_fixture,
    load_json,
    smoke_run,
    split_records,
)


class TestFixture(unittest.TestCase):
    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        self.assertEqual(len(fixture), 6)
        self.assertLessEqual(len(fixture), MAX_SAMPLES)
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})
            self.assertEqual(rec["pixels"], [0] * 16 if rec["label"] == 0 else diagonal)

    def test_explicit_disjoint_split(self):
        parts = split_records(load_fixture(), load_json("split.json"))
        self.assertEqual({k: [r["record_id"] for r in v] for k, v in parts.items()}, {
            "train": ["r001", "r004"],
            "validation": ["r002", "r005"],
            "test": ["r003", "r006"],
        })


class TestSmokeRun(unittest.TestCase):
    def setUp(self):
        self.result = smoke_run(load_fixture(), load_json("split.json"))

    def test_recorded_step_count_within_budget(self):
        self.assertEqual(self.result["epochs"], 1)
        self.assertEqual(self.result["steps"], 2)
        self.assertEqual(len(self.result["step_losses"]), 2)
        self.assertLessEqual(self.result["steps"], MAX_STEPS)

    def test_outputs_and_losses_are_finite(self):
        values = (self.result["step_losses"] + self.result["test_probabilities"]
                  + [self.result["train_loss_after"], self.result["validation_loss"]])
        self.assertTrue(all(math.isfinite(v) for v in values))

    def test_prediction_shape_and_range(self):
        self.assertEqual(self.result["test_record_ids"], ["r003", "r006"])
        self.assertEqual(self.result["test_prediction_shape"], [2])
        self.assertEqual(len(self.result["test_probabilities"]), 2)
        self.assertTrue(all(0.0 < p < 1.0 for p in self.result["test_probabilities"]))

    def test_concrete_values_for_seed_0(self):
        self.assertEqual(self.result["train_order"], ["r001", "r004"])
        self.assertAlmostEqual(self.result["step_losses"][0], math.log(2), places=12)
        self.assertAlmostEqual(self.result["validation_loss"], 0.4906633070255533, places=12)
        self.assertAlmostEqual(self.result["test_probabilities"][0], 0.5075203279027378, places=12)
        self.assertAlmostEqual(self.result["test_probabilities"][1], 0.7610741259387875, places=12)

    def test_model_is_labelled(self):
        self.assertFalse(self.result["is_constant_model"])
        self.assertIn("logistic regression", self.result["model"])
        self.assertIn("not a constant/naive model", self.result["model"])

    def test_repeated_run_is_identical(self):
        self.assertEqual(smoke_run(load_fixture(), load_json("split.json")), self.result)

    def test_saved_expected_result_matches(self):
        self.assertEqual(load_json("expected_result.json"), self.result)


class TestBudgetAndFailureCases(unittest.TestCase):
    def _many(self, n):
        base = load_fixture()
        records = []
        for i in range(n):
            rec = copy.deepcopy(base[i % 6])
            rec["record_id"] = f"x{i:03d}"
            records.append(rec)
        return records

    def test_more_than_32_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "33 samples > 32"):
            smoke_run(self._many(33), load_json("split.json"))

    def test_more_than_20_steps_rejected(self):
        records = self._many(25)
        for i, rec in enumerate(records):                 # 21 train records, 2 val, 2 test
            rec["group_id"] = "g1" if i < 21 else ("g2v" if i < 23 else "g3")
        split = {"train": ["g1"], "validation": ["g2v"], "test": ["g3"]}
        with self.assertRaisesRegex(ValueError, "21 update steps > 20"):
            smoke_run(records, split)

    def test_more_than_one_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError, "epochs must be 1..1"):
            smoke_run(load_fixture(), load_json("split.json"), epochs=2)

    def test_group_overlap_rejected(self):
        split = load_json("split.json")
        split["test"].append("g1")
        with self.assertRaisesRegex(ValueError, "group_id 'g1' is in both"):
            smoke_run(load_fixture(), split)


if __name__ == "__main__":
    unittest.main()
