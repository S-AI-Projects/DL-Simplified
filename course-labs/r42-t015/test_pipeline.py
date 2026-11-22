"""Unit tests for the explicit image preprocessing pipeline."""

import json
import unittest
from pathlib import Path

import numpy as np

from pipeline import (
    load_fixture,
    preprocess_image,
    transform_split,
    validate_records,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(FIXTURE_PATH)
        cls.records = cls.fixture["records"]

    def test_fixture_contains_six_records(self):
        self.assertEqual(len(self.records), 6)

    def test_preprocessed_shape_and_dtype(self):
        image = preprocess_image(self.records[0]["pixels"])
        self.assertEqual(image.shape, (4, 4, 1))
        self.assertEqual(image.dtype, np.float32)

    def test_pixel_values_are_normalized(self):
        pixels = [[0, 255, 0, 255]] * 4
        image = preprocess_image(pixels)
        self.assertEqual(float(image[0, 0, 0]), 0.0)
        self.assertEqual(float(image[0, 1, 0]), 1.0)
        self.assertTrue(np.all(image >= 0.0))
        self.assertTrue(np.all(image <= 1.0))

    def test_diagonal_is_normalized_correctly(self):
        image = preprocess_image(self.records[1]["pixels"])
        self.assertTrue(np.all(np.diag(image[:, :, 0]) == 1.0))
        self.assertEqual(float(image[0, 1, 0]), 0.0)

    def test_invalid_image_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_image([[0, 0], [0, 0]])

    def test_non_numeric_pixels_are_rejected(self):
        pixels = [[0, 0, 0, 0] for _ in range(4)]
        pixels[0][0] = "white"
        with self.assertRaises(ValueError):
            preprocess_image(pixels)

    def test_boolean_pixels_are_rejected(self):
        pixels = [[0, 0, 0, 0] for _ in range(4)]
        pixels[0][0] = True
        with self.assertRaises(ValueError):
            preprocess_image(pixels)

    def test_out_of_range_pixels_are_rejected(self):
        for invalid_value in (-1, 256):
            pixels = [[0, 0, 0, 0] for _ in range(4)]
            pixels[0][0] = invalid_value
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    preprocess_image(pixels)

    def test_non_finite_pixels_are_rejected(self):
        pixels = [[0, 0, 0, 0] for _ in range(4)]
        pixels[0][0] = float("nan")
        with self.assertRaises(ValueError):
            preprocess_image(pixels)

    def test_duplicate_record_ids_are_rejected(self):
        records = [dict(record) for record in self.records]
        records[1]["record_id"] = records[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            validate_records(records)

    def test_group_leakage_across_splits_is_rejected(self):
        records = [dict(record) for record in self.records]
        records[2]["group_id"] = records[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            validate_records(records)

    def test_missing_split_is_rejected(self):
        records = [
            dict(record) for record in self.records
            if record["split"] != "test"
        ]
        with self.assertRaisesRegex(ValueError, "missing required splits"):
            validate_records(records)

    def test_each_split_has_expected_output(self):
        for split in ("train", "validation", "test"):
            images, labels = transform_split(self.records, split)
            with self.subTest(split=split):
                self.assertEqual(images.shape, (2, 4, 4, 1))
                self.assertEqual(images.dtype, np.float32)
                self.assertEqual(labels.tolist(), [0, 1])

    def test_transformation_is_stateless_across_splits(self):
        train_images, _ = transform_split(self.records, "train")
        validation_images, _ = transform_split(self.records, "validation")
        test_images, _ = transform_split(self.records, "test")

        np.testing.assert_array_equal(train_images[0], validation_images[0])
        np.testing.assert_array_equal(train_images[1], test_images[1])

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            transform_split(self.records, "holdout")

    def test_contaminated_fixture_is_rejected(self):
        contaminated = json.loads(json.dumps(self.fixture))
        contaminated["records"][4]["group_id"] = (
            contaminated["records"][0]["group_id"]
        )

        with self.assertRaisesRegex(ValueError, "appears in both"):
            validate_records(contaminated["records"])


if __name__ == "__main__":
    unittest.main()
