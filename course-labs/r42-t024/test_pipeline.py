import unittest
from pipeline import compute_binary_metrics

class TestMetrics(unittest.TestCase):
    def test_standard_case(self):
        actuals = [1, 1, 0, 0]
        preds =   [1, 0, 0, 1]
        m = compute_binary_metrics(actuals, preds)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)

    def test_zero_denominator_recall(self):
        actuals = [0, 0, 0]
        preds =   [0, 0, 0]
        m = compute_binary_metrics(actuals, preds, zero_division=0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["accuracy"], 1.0)

    def test_deliberate_failure(self):
        actuals = [1, 0]
        preds = [1, 0]
        m = compute_binary_metrics(actuals, preds)
        with self.assertRaises(AssertionError):
            self.assertEqual(m["accuracy"], 0.0)

if __name__ == "__main__":
    unittest.main()
