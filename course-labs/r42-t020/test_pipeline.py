
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import (
    build_pipeline,
    load_pipeline,
    predict_next,
    save_pipeline,
    validate_records,
)

BASE = Path(__file__).parent


class TestSavedPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(BASE / "fixture.json", encoding="utf-8") as file:
            cls.fixture = json.load(file)

    def test_fixture_has_exact_six_records(self):
        self.assertEqual(len(self.fixture["records"]), 6)
        self.assertEqual(
            [r["record_id"] for r in self.fixture["records"]],
            ["t1", "t2", "t3", "t4", "t5", "t6"]
        )

    def test_fixture_texts_and_labels(self):
        self.assertEqual(
            [r["text"] for r in self.fixture["records"]],
            ["red apple", "green apple", "blue sky",
             "dark sky", "red fruit", "blue cloud"]
        )
        self.assertEqual(
            [r["label"] for r in self.fixture["records"]],
            [1, 1, 0, 0, 1, 0]
        )

    def test_splits_are_disjoint_and_complete(self):
        splits = self.fixture["splits"]
        train = set(splits["train"])
        validation = set(splits["validation"])
        test = set(splits["test"])

        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(
            train | validation | test,
            {f"t{i}" for i in range(1, 7)}
        )

    def test_training_uses_only_training_records(self):
        train_ids = set(self.fixture["splits"]["train"])
        train_records = [
            r for r in self.fixture["records"]
            if r["record_id"] in train_ids
        ]
        model = build_pipeline(train_records)

        self.assertEqual(model["next_words"], {
            "red": "apple",
            "green": "apple"
        })

    def test_save_reload_predictions_match(self):
        train_ids = set(self.fixture["splits"]["train"])
        records = [
            r for r in self.fixture["records"]
            if r["record_id"] in train_ids
        ]
        model = build_pipeline(records)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved_model.json"
            before = predict_next(model, "red")
            save_pipeline(model, path)
            loaded = load_pipeline(path)
            after = predict_next(loaded, "red")

        self.assertEqual(before, "apple")
        self.assertEqual(after, "apple")
        self.assertEqual(before, after)

    def test_unknown_word_returns_unknown_token(self):
        model = build_pipeline(self.fixture["records"][:2])
        self.assertEqual(predict_next(model, "unknown"), "<UNK>")

    def test_empty_input_is_rejected(self):
        model = build_pipeline(self.fixture["records"][:2])
        with self.assertRaises(ValueError):
            predict_next(model, " ")

    def test_invalid_record_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_records([{"record_id": "x", "text": "hello"}])

    def test_unsupported_schema_version_is_rejected(self):
        model = build_pipeline(self.fixture["records"][:2])
        model["schema_version"] = "99.0"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_model.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump(model, file)

            with self.assertRaisesRegex(ValueError, "Unsupported schema version"):
                load_pipeline(path)

    def test_missing_mapping_is_rejected(self):
        model = build_pipeline(self.fixture["records"][:2])
        del model["next_words"]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_model.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump(model, file)

            with self.assertRaisesRegex(ValueError, "next_words"):
                load_pipeline(path)


if __name__ == "__main__":
    unittest.main()

