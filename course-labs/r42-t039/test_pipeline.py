import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import compute_record_representations

class TestPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_record_order_invariance(self):
        rep1 = compute_record_representations(self.fixture)
        shuffled = list(reversed(self.fixture))
        rep2 = compute_record_representations(shuffled)
        self.assertEqual(rep1, rep2)

    def test_shape_mismatch_rejected(self):
        bad_fixture = [
            {"record_id": "bad_shape", "pixels": [[0, 0], [0, 0]], "label": 0}
        ]
        with self.assertRaises(ValueError):
            compute_record_representations(bad_fixture)

if __name__ == "__main__":
    unittest.main()
