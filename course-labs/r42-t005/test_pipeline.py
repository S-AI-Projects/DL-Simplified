"""
Unit tests for R42-T005 — audio preprocessing contract.

Offline, CPU-only, standard library only, deterministic (no randomness, so no
seed is needed), six synthetic eight-sample waveforms with a fixed
train/validation/test split.

Run with:
    python3 -m unittest discover -s course-labs/r42-t005 -p "test_*.py"
"""

import copy
import math
import unittest

from pipeline import (
    SplitDataset,
    WaveformPreprocessor,
    load_fixture,
    load_json,
    run_preprocessing,
    shape_of,
)

R2 = math.sqrt(2.0)          # 1 / std, std = sqrt(0.5)
ALT = [-1, 1, -1, 1, -1, 1, -1, 1]


class TestFixture(unittest.TestCase):
    def test_fixture_matches_task_description(self):
        fixture = load_fixture()
        self.assertEqual(len(fixture), 6)
        self.assertEqual(len({r["record_id"] for r in fixture}), 6)
        for rec in fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "sample_rate", "samples", "label"})
            self.assertEqual(rec["sample_rate"], 8000)
            self.assertEqual(rec["samples"], [0] * 8 if rec["label"] == 0 else ALT)
        self.assertEqual(sorted(r["label"] for r in fixture), [0, 0, 0, 1, 1, 1])


class TestOutputs(unittest.TestCase):
    """Output values, dtype and shape for the fixture."""

    def setUp(self):
        self.result = run_preprocessing(load_fixture(), load_json("split.json"))

    def test_fitted_on_train_only(self):
        self.assertEqual(self.result["fitted_on"], ["a001", "a004"])
        self.assertEqual(self.result["train_mean"], 0.0)
        self.assertAlmostEqual(self.result["train_std"], math.sqrt(0.5), places=15)

    def test_shape_is_channels_by_time(self):
        for split in ("train", "validation", "test"):
            for array in self.result["outputs"][split].values():
                self.assertEqual(shape_of(array), (1, 8))

    def test_dtype_is_float(self):
        for split in ("train", "validation", "test"):
            for array in self.result["outputs"][split].values():
                self.assertTrue(all(type(v) is float for v in array[0]))
        self.assertEqual(self.result["output_dtype"], "float64")

    def test_exact_values_and_time_order(self):
        out = self.result["outputs"]
        self.assertEqual(out["test"]["a003"], [[0.0] * 8])
        for got, exp in zip(out["test"]["a006"][0], [s * R2 for s in ALT]):
            self.assertAlmostEqual(got, exp, places=12)
        self.assertLess(out["test"]["a006"][0][0], 0)       # time order kept: starts with -1
        self.assertGreater(out["test"]["a006"][0][1], 0)

    def test_split_record_ids(self):
        ids = {k: sorted(v) for k, v in self.result["outputs"].items()}
        self.assertEqual(ids, {"train": ["a001", "a004"],
                               "validation": ["a002", "a005"],
                               "test": ["a003", "a006"]})

    def test_saved_expected_result_matches(self):
        saved = load_json("expected_result.json")
        self.assertEqual(saved["contaminated_split"],
                         "rejected: contamination: group_id 'g2' is in both 'train' and 'validation'")
        del saved["contaminated_split"]
        self.assertEqual(saved, self.result)


class TestNoHeldOutFitting(unittest.TestCase):
    def test_fit_rejects_raw_record_lists(self):
        with self.assertRaisesRegex(TypeError, "only accepts a SplitDataset"):
            WaveformPreprocessor().fit(load_fixture())

    def test_held_out_values_do_not_change_fit(self):
        records = copy.deepcopy(load_fixture())
        for rec in records:
            if rec["group_id"] in ("g2", "g3"):
                rec["samples"] = [100] * 8
        pre = WaveformPreprocessor().fit(SplitDataset(records, load_json("split.json")))
        self.assertEqual(pre.mean, 0.0)
        self.assertAlmostEqual(pre.std, math.sqrt(0.5), places=15)

    def test_contamination_fixture_rejected(self):
        with self.assertRaisesRegex(ValueError, "contamination: group_id 'g2' is in both 'train' and 'validation'"):
            SplitDataset(load_fixture(), load_json("contaminated_split.json"))

    def test_duplicate_record_id_rejected(self):
        records = load_fixture() + [copy.deepcopy(load_fixture()[0])]
        with self.assertRaisesRegex(ValueError, "contamination: duplicate record_id"):
            SplitDataset(records, load_json("split.json"))


class TestInputContract(unittest.TestCase):
    def setUp(self):
        self.pre = WaveformPreprocessor().fit(SplitDataset(load_fixture(), load_json("split.json")))
        self.rec = copy.deepcopy(load_fixture()[3])

    def test_mismatched_sample_rate_rejected(self):
        self.rec["sample_rate"] = 16000
        with self.assertRaisesRegex(ValueError, "sample_rate mismatch, expected 8000, got 16000"):
            self.pre.transform(self.rec)

    def test_wrong_length_rejected_not_padded(self):
        self.rec["samples"] = ALT[:6]
        with self.assertRaisesRegex(ValueError, "expected 8 samples, got 6"):
            self.pre.transform(self.rec)

    def test_multichannel_input_rejected(self):
        self.rec["samples"] = [ALT, ALT]
        with self.assertRaisesRegex(ValueError, "expected mono 1-D samples"):
            self.pre.transform(self.rec)

    def test_non_numeric_sample_rejected(self):
        self.rec["samples"] = ALT[:7] + ["x"]
        with self.assertRaisesRegex(ValueError, "not a finite number"):
            self.pre.transform(self.rec)

    def test_zero_variance_training_rejected(self):
        records = copy.deepcopy(load_fixture())
        for rec in records:
            rec["samples"] = [0] * 8
        with self.assertRaisesRegex(ValueError, "zero variance"):
            WaveformPreprocessor().fit(SplitDataset(records, load_json("split.json")))

    def test_transform_before_fit_rejected(self):
        with self.assertRaises(RuntimeError):
            WaveformPreprocessor().transform(self.rec)


if __name__ == "__main__":
    unittest.main()
