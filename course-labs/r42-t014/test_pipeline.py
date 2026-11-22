
import unittest

from pipeline import (
    classification_metrics,
    mean_absolute_error,
    load_fixture,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_precision_and_recall(self):
        result = classification_metrics(
            [1, 1, 0, 0], [1, 0, 1, 0]
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_zero_denominator_returns_zero(self):
        result = classification_metrics([0, 0], [0, 0])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_unequal_lengths_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([1], [1, 0])

    def test_non_binary_values_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 2], [0, 1])


class TestRegressionMetric(unittest.TestCase):
    def test_mae(self):
        self.assertAlmostEqual(
            mean_absolute_error([1, 2, 3], [1, 3, 2]),
            2 / 3
        )

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            mean_absolute_error([], [])

    def test_unequal_lengths_rejected(self):
        with self.assertRaises(ValueError):
            mean_absolute_error([1], [1, 2])


class TestFixture(unittest.TestCase):
    def test_fixture_contract(self):
        fixture = load_fixture()
        self.assertEqual(len(fixture["records"]), 6)
        self.assertEqual(fixture["sample_rate"], 8000)

    def test_record_ids_unique(self):
        fixture = load_fixture()
        ids = [record["record_id"] for record in fixture["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_waveform_length(self):
        fixture = load_fixture()
        for record in fixture["records"]:
            self.assertEqual(len(record["samples"]), 8)


if __name__ == "__main__":
    unittest.main()
