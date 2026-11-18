import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import ColourisationPipeline

class TestSavedPipeline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_save_and_reload_identity(self):
        pipe = ColourisationPipeline(scale_factor=0.8)
        preds_before = pipe.predict(self.fixture)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pipe.save_state(tmp_path)
            reloaded = ColourisationPipeline.load_state(tmp_path)
            preds_after = reloaded.predict(self.fixture)

            for b_img, a_img in zip(preds_before, preds_after):
                for b_row, a_row in zip(b_img, a_img):
                    for b_val, a_val in zip(b_row, a_row):
                        self.assertAlmostEqual(b_val, a_val, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_version_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({"version": "9.9", "scale_factor": 1.0}, tmp)
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                ColourisationPipeline.load_state(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
