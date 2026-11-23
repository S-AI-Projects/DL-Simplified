"""
Unit tests for R42-T002 — NEU steel surface defects CNN, leakage-free split.

Offline, CPU-only, standard library only, fixed seed (42), six-record fixture.
The check exits non-zero when an expected result is deliberately changed.

Run with:
    python3 -m unittest discover -s course-labs/r42-t002 -p "test_*.py"
"""

import copy
import json
import pathlib
import unittest

from pipeline import (
    LAB_DIR,
    SPLITS,
    load_fixture,
    make_split_manifest,
    manifest_to_json,
    validate_manifest,
)

# Concrete expected result for the fixture with seed 42.
EXPECTED_SPLITS = {
    "train": ["r002", "r005"],
    "validation": ["r001", "r004"],
    "test": ["r003", "r006"],
}
EXPECTED_GROUPS = {"train": ["g2"], "validation": ["g1"], "test": ["g3"]}


class TestFixture(unittest.TestCase):
    """The literal synthetic fixture matches the task description."""

    def setUp(self):
        self.fixture = load_fixture()

    def test_six_records_with_schema_fields(self):
        self.assertEqual(len(self.fixture), 6)
        for rec in self.fixture:
            self.assertEqual(set(rec), {"record_id", "group_id", "pixels", "label"})

    def test_three_all_zero_label0_and_three_diagonal_label1(self):
        zero = [0] * 16
        diagonal = [255 if r == c else 0 for r in range(4) for c in range(4)]
        label0 = [r for r in self.fixture if r["label"] == 0]
        label1 = [r for r in self.fixture if r["label"] == 1]
        self.assertEqual(len(label0), 3)
        self.assertEqual(len(label1), 3)
        self.assertTrue(all(r["pixels"] == zero for r in label0))
        self.assertTrue(all(r["pixels"] == diagonal for r in label1))

    def test_unique_record_ids_and_declared_groups(self):
        ids = [r["record_id"] for r in self.fixture]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({r["group_id"] for r in self.fixture}, {"g1", "g2", "g3"})


class TestSplitManifest(unittest.TestCase):
    """Concrete result, disjointness, group isolation and determinism."""

    def setUp(self):
        self.fixture = load_fixture()
        self.manifest = make_split_manifest(self.fixture, seed=42)

    def test_exact_record_ids_per_split(self):
        self.assertEqual(self.manifest["splits"], EXPECTED_SPLITS)

    def test_exact_groups_per_split(self):
        self.assertEqual(self.manifest["groups"], EXPECTED_GROUPS)

    def test_train_validation_test_are_disjoint(self):
        train = set(self.manifest["splits"]["train"])
        val = set(self.manifest["splits"]["validation"])
        test = set(self.manifest["splits"]["test"])
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_every_record_assigned_exactly_once(self):
        all_ids = [rid for s in SPLITS for rid in self.manifest["splits"][s]]
        self.assertEqual(sorted(all_ids), sorted(r["record_id"] for r in self.fixture))

    def test_no_group_in_more_than_one_split(self):
        group_of = {r["record_id"]: r["group_id"] for r in self.fixture}
        seen = {}
        for split in SPLITS:
            for rid in self.manifest["splits"][split]:
                self.assertEqual(seen.setdefault(group_of[rid], split), split)

    def test_repeated_runs_give_identical_manifest(self):
        runs = {manifest_to_json(make_split_manifest(load_fixture(), seed=42))
                for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_record_order_does_not_change_manifest(self):
        reversed_fixture = list(reversed(self.fixture))
        self.assertEqual(make_split_manifest(reversed_fixture, seed=42), self.manifest)

    def test_saved_expected_manifest_matches(self):
        saved = json.loads((LAB_DIR / "expected_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.manifest)


class TestLeakageRejected(unittest.TestCase):
    """Intentionally overlapping IDs/groups and bad inputs are rejected."""

    def setUp(self):
        self.fixture = load_fixture()
        self.manifest = make_split_manifest(self.fixture, seed=42)

    def test_overlapping_record_id_between_splits_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["splits"]["test"].append("r002")          # r002 is already in train
        with self.assertRaisesRegex(ValueError, "'r002' appears in both 'train' and 'test'"):
            validate_manifest(bad, self.fixture)

    def test_group_split_across_two_splits_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["splits"]["train"].remove("r005")         # move half of g2 to test
        bad["splits"]["test"].append("r005")
        with self.assertRaisesRegex(ValueError, "group_id 'g2' appears in both"):
            validate_manifest(bad, self.fixture)

    def test_duplicate_record_id_in_input_rejected(self):
        records = self.fixture + [copy.deepcopy(self.fixture[0])]
        with self.assertRaisesRegex(ValueError, "duplicate record_id 'r001'"):
            make_split_manifest(records)

    def test_missing_group_id_rejected(self):
        records = copy.deepcopy(self.fixture)
        del records[2]["group_id"]
        with self.assertRaisesRegex(ValueError, "missing or invalid group_id"):
            make_split_manifest(records)

    def test_unassigned_record_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["splits"]["validation"].remove("r004")
        bad["splits"]["validation"].remove("r001")
        with self.assertRaisesRegex(ValueError, "not assigned to any split"):
            validate_manifest(bad, self.fixture)

    def test_too_few_groups_rejected(self):
        records = copy.deepcopy(self.fixture)
        for rec in records:
            rec["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "need at least 3 groups"):
            make_split_manifest(records)


class TestChronologicalSplit(unittest.TestCase):
    """With timestamps (test-only variant of the fixture), splits are chronological."""

    def setUp(self):
        self.records = copy.deepcopy(load_fixture())
        # g3 earliest, then g1, then g2 (day numbers; simulated)
        day = {"r003": 1, "r006": 2, "r001": 3, "r004": 4, "r002": 5, "r005": 6}
        for rec in self.records:
            rec["timestamp"] = day[rec["record_id"]]

    def test_oldest_group_train_newest_group_test(self):
        m = make_split_manifest(self.records)
        self.assertEqual(m["strategy"], "chronological-group")
        self.assertEqual(m["groups"], {"train": ["g3"], "validation": ["g1"], "test": ["g2"]})

    def test_overlapping_time_ranges_rejected(self):
        self.records[5]["timestamp"] = 0              # r006 (g3) now oldest...
        self.records[0]["timestamp"] = 10             # ...and r001 (g1) newest
        with self.assertRaisesRegex(ValueError, "not chronological"):
            make_split_manifest(self.records)


if __name__ == "__main__":
    unittest.main()
