"""Unit tests for R42-T045 explicit preprocessing."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import preprocess_dataset, preprocess_image


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_fixture_preprocessing_shape_and_range(self):
        """Assert output shape (4, 4, 1) and values scaled strictly to [0.0, 1.0]."""
        processed = preprocess_dataset(self.fixture)
        self.assertEqual(len(processed), 6)

        for rec in processed:
            tensor = rec["tensor"]
            # Height = 4
            self.assertEqual(len(tensor), 4)
            for row in tensor:
                # Width = 4
                self.assertEqual(len(row), 4)
                for channel in row:
                    # Explicit channel axis = 1
                    self.assertEqual(len(channel), 1)
                    val = channel[0]
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

        # Label 0 all-zero scaled is 0.0
        self.assertEqual(processed[0]["tensor"][0][0][0], 0.0)
        # Label 1 white diagonal scaled is 1.0
        self.assertEqual(processed[3]["tensor"][0][0][0], 1.0)
        self.assertEqual(processed[3]["tensor"][0][1][0], 0.0)

    def test_rejection_of_out_of_range_pixels(self):
        """Values < 0 or > 255 must be rejected."""
        bad_pixels = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ValueError) as ctx:
            preprocess_image(bad_pixels)
        self.assertIn("out of range", str(ctx.exception))

    def test_rejection_of_invalid_dimensions(self):
        """Non-4x4 dimensions must be rejected."""
        bad_shape = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError) as ctx:
            preprocess_image(bad_shape)
        self.assertIn("Expected 4 rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
