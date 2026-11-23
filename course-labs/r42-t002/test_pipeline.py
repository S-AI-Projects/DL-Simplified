"""
Unit tests for R42-T002 — NEU Steel Surface Defects CNN leakage-free split.

Runs offline, CPU-only, with a fixed fixture (no randomness needed).
A test must exit non-zero when an expected result is deliberately changed.

Run with:
    python3 -m unittest discover -s course-labs/r42-t002 -p "test_*.py"
"""

import copy
import unittest

from pipeline import (
    make_fixture,
    build_split_manifest,
    validate_manifest,
    manifest_fingerprint,
    _DETERMINISTIC_GROUP_SPLIT,
)


class TestManifestStructure(unittest.TestCase):
    """Basic structure of the split manifest."""

    def setUp(self):
        self.fixture = make_fixture()
        self.manifest = build_split_manifest(self.fixture)

    def test_manifest_has_three_splits(self):
        self.assertEqual(set(self.manifest.keys()), {"train", "val", "test"})

    def test_all_splits_non_empty(self):
        for split, ids in self.manifest.items():
            self.assertGreater(len(ids), 0, f"Split {split!r} must not be empty")

    def test_total_records_equal_fixture_size(self):
        total = sum(len(ids) for ids in self.manifest.values())
        self.assertEqual(total, len(self.fixture))

    def test_all_record_ids_covered(self):
        all_ids = {r["record_id"] for r in self.fixture}
        assigned = {rid for ids in self.manifest.values() for rid in ids}
        self.assertEqual(all_ids, assigned)


class TestDisjointSplits(unittest.TestCase):
    """No record may appear in more than one split."""

    def test_no_overlap_between_splits(self):
        fixture = make_fixture()
        manifest = build_split_manifest(fixture)
        splits = list(manifest.values())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = set(splits[i]) & set(splits[j])
                self.assertEqual(overlap, set(), f"Overlap between split {i} and {j}: {overlap}")

    def test_validate_manifest_reports_no_errors_on_clean_fixture(self):
        fixture = make_fixture()
        manifest = build_split_manifest(fixture)
        all_ids = [r["record_id"] for r in fixture]
        errors = validate_manifest(manifest, all_ids)
        self.assertEqual(errors, [], msg=f"Unexpected errors: {errors}")


class TestGroupIsolation(unittest.TestCase):
    """All records sharing a group_id must land in the same split."""

    def test_group_isolation(self):
        fixture = make_fixture()
        manifest = build_split_manifest(fixture)

        # Build reverse map: record_id → split
        rid_to_split = {}
        for split, ids in manifest.items():
            for rid in ids:
                rid_to_split[rid] = split

        # Build group → record mapping from fixture
        group_to_rids: dict = {}
        for rec in fixture:
            group_to_rids.setdefault(rec["group_id"], []).append(rec["record_id"])

        for gid, rids in group_to_rids.items():
            splits_for_group = {rid_to_split[rid] for rid in rids}
            self.assertEqual(
                len(splits_for_group), 1,
                msg=f"Group {gid!r} spans multiple splits: {splits_for_group}",
            )


class TestDeterminism(unittest.TestCase):
    """Repeated calls must produce identical manifests."""

    def test_repeated_runs_same_manifest(self):
        fixture = make_fixture()
        m1 = build_split_manifest(fixture)
        m2 = build_split_manifest(fixture)
        self.assertEqual(m1, m2)

    def test_fingerprint_stable(self):
        fixture = make_fixture()
        fp1 = manifest_fingerprint(build_split_manifest(fixture))
        fp2 = manifest_fingerprint(build_split_manifest(fixture))
        self.assertEqual(fp1, fp2)

    def test_fingerprint_known_value(self):
        """
        Lock the fingerprint to a known value.
        If the split logic changes, this test MUST fail — intentional guard.
        Change only when the split design is deliberately updated.
        """
        import json, hashlib
        # Build expected manifest directly (bypassing the function under test)
        expected_manifest = {
            "train": ["r001", "r002"],
            "val":   ["r003", "r004"],
            "test":  ["r005", "r006"],
        }
        canonical = json.dumps(expected_manifest, sort_keys=True, separators=(",", ":"))
        expected_fp = hashlib.sha256(canonical.encode()).hexdigest()

        fixture = make_fixture()
        actual_fp = manifest_fingerprint(build_split_manifest(fixture))
        self.assertEqual(actual_fp, expected_fp)


class TestManifestValidation(unittest.TestCase):
    """validate_manifest must catch deliberate leakage injections."""

    def test_overlapping_ids_detected(self):
        """Inject a duplicate across splits → validation must catch it."""
        bad_manifest = {
            "train": ["r001", "r002", "r003"],   # r003 also in val
            "val":   ["r003", "r004"],
            "test":  ["r005", "r006"],
        }
        fixture = make_fixture()
        all_ids = [r["record_id"] for r in fixture]
        errors = validate_manifest(bad_manifest, all_ids)
        self.assertGreater(len(errors), 0, "Expected leakage errors to be reported")

    def test_missing_record_detected(self):
        """Drop a record from the manifest → validation must catch it."""
        bad_manifest = {
            "train": ["r001"],         # r002 missing
            "val":   ["r003", "r004"],
            "test":  ["r005", "r006"],
        }
        fixture = make_fixture()
        all_ids = [r["record_id"] for r in fixture]
        errors = validate_manifest(bad_manifest, all_ids)
        self.assertGreater(len(errors), 0, "Expected missing-record error")


class TestNonZeroCountSentinel(unittest.TestCase):
    """Suite must have a non-zero test count and sentinel guard."""

    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        count = suite.countTestCases()
        self.assertGreater(count, 0)

    def test_sentinel_outcome_guard(self):
        """Do not alter the expected value — deliberate change must fail this test."""
        fixture = make_fixture()
        manifest = build_split_manifest(fixture)
        all_ids = [r["record_id"] for r in fixture]
        errors = validate_manifest(manifest, all_ids)
        outcome = "PASS" if not errors else "FAIL"
        self.assertEqual(outcome, "PASS")


if __name__ == "__main__":
    unittest.main()
