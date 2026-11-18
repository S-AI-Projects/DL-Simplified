"""Unit tests for R42-T049 input perturbation robustness."""

import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import classify_record, run_pipeline_with_order_invariance


class TestPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_record_order_permutation_invariance(self):
        """Permuting input record order must preserve identical ID-aligned predictions."""
        original_preds = run_pipeline_with_order_invariance(self.fixture)

        # Shuffle records with fixed seed
        shuffled_fixture = list(self.fixture)
        random.seed(42)
        random.shuffle(shuffled_fixture)

        shuffled_preds = run_pipeline_with_order_invariance(shuffled_fixture)
        self.assertEqual(original_preds, shuffled_preds)

    def test_unsupported_shape_rejection(self):
        """Unsupported image dimensions must be rejected with ValueError."""
        bad_shape_rec = {
            "record_id": "bad_shape",
            "group_id": "grp_x",
            "pixels": [[0, 0], [0, 0]],
            "label": 0
        }
        with self.assertRaises(ValueError) as ctx:
            classify_record(bad_shape_rec)
        self.assertIn("Invalid pixel dimensions", str(ctx.exception))

    def test_missing_field_rejection(self):
        """Record missing required field must be rejected."""
        missing_pixels_rec = {
            "record_id": "missing_px",
            "group_id": "grp_x",
            "label": 0
        }
        with self.assertRaises(ValueError) as ctx:
            classify_record(missing_pixels_rec)
        self.assertIn("missing required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
