import json
import unittest
from pathlib import Path

from pipeline import validate_record, validate_records


FIXTURE_PATH = Path(__file__).parent / "fixture.json"


class TestPipeline(unittest.TestCase):

    def setUp(self):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as file:
            self.records = json.load(file)

    def test_six_valid_fixture_records(self):
        self.assertEqual(len(self.records), 6)
        self.assertEqual(validate_records(self.records), [])

    def test_missing_record_id(self):
        record = {"text": "red apple", "label": 1}
        errors = validate_record(record)
        self.assertTrue(any("record_id" in error for error in errors))

    def test_invalid_text(self):
        record = {"record_id": "t1", "text": 123, "label": 1}
        errors = validate_record(record)
        self.assertTrue(any("text" in error for error in errors))

    def test_invalid_numeric_label(self):
        record = {"record_id": "t1", "text": "red apple", "label": "wrong"}
        errors = validate_record(record)
        self.assertTrue(any("label" in error for error in errors))

    def test_tensor_like_label_is_rejected(self):
        record = {"record_id": "t1", "text": "red apple", "label": [1, 0]}
        errors = validate_record(record)
        self.assertTrue(any("label" in error for error in errors))

    def test_duplicate_record_id(self):
        records = [
            {"record_id": "t1", "text": "red apple", "label": 1},
            {"record_id": "t1", "text": "green apple", "label": 1},
        ]
        errors = validate_records(records)
        self.assertTrue(any("duplicate" in error for error in errors))


if __name__ == "__main__":
    unittest.main()