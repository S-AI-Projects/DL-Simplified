import unittest
import tempfile
import os
from pipeline import SavedPipeline

class TestSavedPipeline(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            {"record_id": "img_001", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
        ]

    def test_save_and_reload(self):
        pipe = SavedPipeline(threshold=150, version="1.0.0")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pipe.save_config(tmp_path)
            reloaded = SavedPipeline.load_config(tmp_path)
            self.assertEqual(pipe.threshold, reloaded.threshold)
            self.assertEqual(pipe.version, reloaded.version)

            for rec in self.fixture:
                self.assertEqual(pipe.predict(rec), reloaded.predict(rec))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_schema_mismatch(self):
        pipe = SavedPipeline()
        with self.assertRaises(ValueError):
            pipe.predict({"pixels": [0]*5})

    def test_deliberate_failure(self):
        pipe = SavedPipeline()
        with self.assertRaises(AssertionError):
            self.assertEqual(pipe.threshold, 99999)

if __name__ == "__main__":
    unittest.main()
