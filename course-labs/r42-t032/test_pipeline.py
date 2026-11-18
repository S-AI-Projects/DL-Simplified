import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import split_manifest

class TestSplitManifest(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_disjoint_splits(self):
        manifest = split_manifest(self.fixture)
        train_set = set(manifest["train"])
        test_set = set(manifest["test"])
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertEqual(len(train_set) + len(test_set), 6)

    def test_determinism(self):
        m1 = split_manifest(self.fixture)
        m2 = split_manifest(self.fixture)
        self.assertEqual(m1, m2)

    def test_group_isolation(self):
        manifest = split_manifest(self.fixture)
        id_to_group = {r["record_id"]: r["group_id"] for r in self.fixture}
        train_groups = {id_to_group[rid] for rid in manifest["train"]}
        test_groups = {id_to_group[rid] for rid in manifest["test"]}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_duplicate_rejection(self):
        bad_fixture = list(self.fixture) + [self.fixture[0]]
        with self.assertRaises(ValueError):
            split_manifest(bad_fixture)

if __name__ == "__main__":
    unittest.main()
