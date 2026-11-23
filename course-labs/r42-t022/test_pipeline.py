import unittest
from pipeline import create_deterministic_split, verify_manifest

class TestSplitManifest(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            {"record_id": "img_001", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_002", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_003", "group_id": "grp_b", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "group_id": "grp_b", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_005", "group_id": "grp_c", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_006", "group_id": "grp_c", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
        ]

    def test_splits_are_disjoint(self):
        manifest = create_deterministic_split(self.fixture)
        tr, va, te = set(manifest["train"]), set(manifest["val"]), set(manifest["test"])
        self.assertEqual(len(tr & va), 0)
        self.assertEqual(len(tr & te), 0)
        self.assertEqual(len(va & te), 0)
        self.assertTrue(verify_manifest(manifest, self.fixture))

    def test_deterministic_reproducibility(self):
        m1 = create_deterministic_split(self.fixture, seed=42)
        m2 = create_deterministic_split(self.fixture, seed=42)
        self.assertEqual(m1, m2)

    def test_group_isolation(self):
        manifest = create_deterministic_split(self.fixture)
        self.assertEqual(manifest["train"], ["img_001", "img_002"])
        self.assertEqual(manifest["val"], ["img_003", "img_004"])
        self.assertEqual(manifest["test"], ["img_005", "img_006"])

    def test_deliberate_failure(self):
        manifest = create_deterministic_split(self.fixture)
        with self.assertRaises(AssertionError):
            self.assertEqual(len(manifest["train"]), 99)

if __name__ == "__main__":
    unittest.main()
