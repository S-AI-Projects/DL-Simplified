import unittest
from pipeline import SimpleSmokeLinearModel

class TestSmokeRun(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"token_ids": [1, 2], "label": 1},
            {"token_ids": [3, 2], "label": 1},
            {"token_ids": [4, 5], "label": 0},
            {"token_ids": [6, 5], "label": 0},
            {"token_ids": [1, 7], "label": 1},
            {"token_ids": [4, 8], "label": 0},
        ]

    def test_smoke_run_bounds(self):
        model = SimpleSmokeLinearModel(vocab_size=10, seed=42)
        res = model.run_smoke_epoch(self.records, max_steps=4)
        self.assertEqual(res["steps_run"], 4)
        self.assertTrue(res["bounded_cpu_smoke"])

    def test_deterministic_losses(self):
        m1 = SimpleSmokeLinearModel(vocab_size=10, seed=42)
        m2 = SimpleSmokeLinearModel(vocab_size=10, seed=42)
        r1 = m1.run_smoke_epoch(self.records, max_steps=6)
        r2 = m2.run_smoke_epoch(self.records, max_steps=6)
        self.assertEqual(r1["final_loss"], r2["final_loss"])

    def test_deliberate_failure(self):
        model = SimpleSmokeLinearModel(vocab_size=10, seed=42)
        res = model.run_smoke_epoch(self.records, max_steps=2)
        with self.assertRaises(AssertionError):
            self.assertEqual(res["steps_run"], 999)

if __name__ == "__main__":
    unittest.main()
