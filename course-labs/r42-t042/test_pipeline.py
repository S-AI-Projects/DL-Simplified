"""Unit tests for R42-T042 deterministic split manifest."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import create_deterministic_split, validate_split_manifest


class TestSplitManifest(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_fixture_contents(self):
        """Verify the 6 synthetic text fixture items."""
        self.assertEqual(len(self.fixture), 6)
        ids = [r["record_id"] for r in self.fixture]
        self.assertEqual(ids, ["t1", "t2", "t3", "t4", "t5", "t6"])

    def test_disjoint_deterministic_splits(self):
        """Splits must be disjoint, deterministically reproducible across runs."""
        manifest1 = create_deterministic_split(self.fixture)
        manifest2 = create_deterministic_split(self.fixture)
        self.assertEqual(manifest1, manifest2)

        validate_split_manifest(manifest1)

        train_set = set(manifest1["train"])
        val_set = set(manifest1["validation"])
        test_set = set(manifest1["test"])
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))

    def test_rejection_of_overlapping_splits(self):
        """Intentionally overlapping split manifest must be rejected."""
        leaky_manifest = {
            "train": ["t1", "t2", "t3", "t4"],
            "validation": ["t4", "t5"],
            "test": ["t6"]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_split_manifest(leaky_manifest)
        self.assertIn("Split leakage detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
