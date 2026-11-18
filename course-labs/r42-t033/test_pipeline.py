import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import MajorityClassBaseline

class TestMajorityClassBaseline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)
        self.train_data = [r for r in self.fixture if r["group_id"] == "grp_a"]
        self.test_data = [r for r in self.fixture if r["group_id"] == "grp_b"]

    def test_fit_and_predict(self):
        model = MajorityClassBaseline()
        model.fit(self.train_data)
        self.assertEqual(model.majority_label, 0)
        preds = model.predict(self.test_data)
        self.assertEqual(preds, [0, 0, 0])

    def test_held_out_evaluation(self):
        model = MajorityClassBaseline()
        model.fit(self.train_data)
        res = model.evaluate(self.test_data)
        # grp_b has label 1, so majority label 0 achieves 0.0 accuracy on held out test
        self.assertEqual(res["accuracy"], 0.0)
        self.assertEqual(res["total_tested"], 3)

    def test_unfit_predict_raises(self):
        model = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            model.predict(self.test_data)

if __name__ == "__main__":
    unittest.main()
