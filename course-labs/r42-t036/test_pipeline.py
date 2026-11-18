import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import BoundedCPULogisticModel

class TestCPUSmokeRun(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_smoke_run_bounds_and_finite_loss(self):
        model = BoundedCPULogisticModel(seed=42)
        res = model.train_smoke_run(self.fixture, max_steps=10)
        self.assertLessEqual(res["step_count"], 20)
        self.assertEqual(res["step_count"], 10)
        self.assertTrue(res["finite_loss"])

    def test_prediction_shape(self):
        model = BoundedCPULogisticModel(seed=42)
        model.train_smoke_run(self.fixture, max_steps=5)
        preds = model.predict(self.fixture)
        self.assertEqual(len(preds), len(self.fixture))
        for p in preds:
            self.assertTrue(0.0 <= p <= 1.0)

if __name__ == "__main__":
    unittest.main()
