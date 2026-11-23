import unittest
from pipeline import generate_error_report

class TestErrorReport(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            {"record_id": "img_001", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_002", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_003", "group_id": "grp_b", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "group_id": "grp_b", "pixels": [255]*16, "label": 1},
            {"record_id": "img_005", "group_id": "grp_c", "pixels": [255]*16, "label": 1},
            {"record_id": "img_006", "group_id": "grp_c", "pixels": [255]*16, "label": 1},
        ]

    def test_injected_error_identified(self):
        rep = generate_error_report(self.fixture, inject_error_id="img_004")
        self.assertEqual(rep["total_errors"], 1)
        err = rep["error_records"][0]
        self.assertEqual(err["record_id"], "img_004")
        self.assertEqual(err["expected"], 1)
        self.assertEqual(err["predicted"], 0)
        self.assertIn("Injected deliberate ambiguity", err["reason_category"])

    def test_correct_records_omitted_from_errors(self):
        rep = generate_error_report(self.fixture, inject_error_id="img_004")
        error_ids = {e["record_id"] for e in rep["error_records"]}
        self.assertNotIn("img_001", error_ids)
        self.assertNotIn("img_002", error_ids)

    def test_deliberate_failure(self):
        rep = generate_error_report(self.fixture, inject_error_id="img_004")
        with self.assertRaises(AssertionError):
            self.assertEqual(rep["total_errors"], 0)

if __name__ == "__main__":
    unittest.main()
