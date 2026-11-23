import unittest
from pipeline import validate_record, validate_dataset

class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            {"record_id": "img_001", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_002", "group_id": "grp_a", "pixels": [0]*16, "label": 0},
            {"record_id": "img_003", "group_id": "grp_b", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "group_id": "grp_b", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_005", "group_id": "grp_c", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_006", "group_id": "grp_c", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
        ]

    def test_clean_fixture_passes(self):
        res = validate_dataset(self.fixture)
        self.assertTrue(res["valid"])
        self.assertEqual(res["error_count"], 0)

    def test_missing_record_id(self):
        rec = {"group_id": "grp_a", "pixels": [0]*16, "label": 0}
        errs = validate_record(rec, set())
        self.assertTrue(any(f == "record_id" for _, f, _ in errs))

    def test_duplicate_record_id(self):
        seen = {"img_001"}
        rec = {"record_id": "img_001", "group_id": "grp_a", "pixels": [0]*16, "label": 0}
        errs = validate_record(rec, seen)
        self.assertTrue(any("Duplicate" in r for _, _, r in errs))

    def test_malformed_pixels(self):
        rec = {"record_id": "img_err", "group_id": "grp_a", "pixels": [0]*10, "label": 0}
        errs = validate_record(rec, set())
        self.assertTrue(any(f == "pixels" for _, f, _ in errs))

    def test_deliberate_failure_assertion(self):
        res = validate_dataset(self.fixture)
        with self.assertRaises(AssertionError):
            self.assertEqual(res["error_count"], 999)

if __name__ == "__main__":
    unittest.main()
