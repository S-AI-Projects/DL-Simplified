import unittest
from pipeline import MajorityClassBaseline

class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.train_data = [
            {"record_id": "img_001", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_002", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_003", "group_id": "grp_b", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "group_id": "grp_b", "pixels": [255]*16, "label": 1},
        ]
        self.test_data = [
            {"record_id": "img_005", "group_id": "grp_c", "pixels": [255]*16, "label": 1},
            {"record_id": "img_006", "group_id": "grp_c", "pixels": [0]*16, "label": 0},
        ]

    def test_fit_and_predict(self):
        clf = MajorityClassBaseline()
        clf.fit(self.train_data)
        self.assertEqual(clf.majority_label, 0)
        res = clf.evaluate(self.test_data)
        self.assertEqual(res["total_test"], 2)
        self.assertEqual(res["accuracy"], 0.5)

    def test_unfitted_raises(self):
        clf = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            clf.predict(self.test_data)

    def test_deliberate_failure(self):
        clf = MajorityClassBaseline()
        clf.fit(self.train_data)
        with self.assertRaises(AssertionError):
            self.assertEqual(clf.majority_label, 999)

if __name__ == "__main__":
    unittest.main()
