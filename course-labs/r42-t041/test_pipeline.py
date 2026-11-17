"""Unit tests for R42-T041 Hand gesture images - CNN schema validator."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import validate_dataset, validate_record


class TestPipelineSchema(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_fixture_shape_and_content(self):
        """Verify fixture contains exactly 6 records, 3 zero images (label 0), 3 diagonal images (label 1)."""
        self.assertEqual(len(self.fixture), 6)
        label_0 = [r for r in self.fixture if r["label"] == 0]
        label_1 = [r for r in self.fixture if r["label"] == 1]
        self.assertEqual(len(label_0), 3)
        self.assertEqual(len(label_1), 3)

        for r in label_0:
            self.assertEqual(r["pixels"], [[0]*4 for _ in range(4)])
        for r in label_1:
            expected = [[255 if i == j else 0 for j in range(4)] for i in range(4)]
            self.assertEqual(r["pixels"], expected)

    def test_valid_fixture_passes(self):
        """The valid synthetic fixture must pass validation without failures."""
        res = validate_dataset(self.fixture)
        self.assertTrue(res["is_valid"])
        self.assertEqual(len(res["failures"]), 0)
        self.assertEqual(res["total_records"], 6)

    def test_missing_record_id(self):
        """Missing or empty record_id must fail with specific reason."""
        bad_record = {
            "group_id": "grp_a",
            "pixels": [[0]*4 for _ in range(4)],
            "label": 0
        }
        res = validate_record(bad_record, set())
        self.assertTrue(any(f[1] == "record_id" for f in res))

    def test_duplicate_record_id(self):
        """Duplicate record ID must fail with explicit reason without dropping records."""
        bad_dataset = [
            {"record_id": "dup_1", "group_id": "g1", "pixels": [[0]*4 for _ in range(4)], "label": 0},
            {"record_id": "dup_1", "group_id": "g2", "pixels": [[0]*4 for _ in range(4)], "label": 0}
        ]
        res = validate_dataset(bad_dataset)
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("Duplicate record ID" in f[2] for f in res["failures"]))

    def test_malformed_pixels_shape(self):
        """Pixels of wrong dimension must fail."""
        bad_record = {
            "record_id": "rec_bad_dim",
            "group_id": "grp_x",
            "pixels": [[0, 0], [0, 0]],
            "label": 0
        }
        res = validate_record(bad_record, set())
        self.assertTrue(any(f[1] == "pixels" for f in res))

    def test_malformed_label(self):
        """Non-binary label must fail."""
        bad_record = {
            "record_id": "rec_bad_label",
            "group_id": "grp_x",
            "pixels": [[0]*4 for _ in range(4)],
            "label": "gesture_peace"
        }
        res = validate_record(bad_record, set())
        self.assertTrue(any(f[1] == "label" for f in res))


if __name__ == "__main__":
    unittest.main()
