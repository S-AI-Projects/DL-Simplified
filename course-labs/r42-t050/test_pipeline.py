"""Unit tests for R42-T050 saved pipeline interface."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import SavedMLPPipeline


class TestSavedPipeline(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)

    def test_save_and_reload_predictions_agreement(self):
        """Predictions before and after save/reload must agree within explicit tolerance."""
        # Non-trivial diagonal-detector weights
        weights = [1.0 if (i % 5 == 0) else -0.5 for i in range(16)]
        model = SavedMLPPipeline(weights=weights, bias=-0.2)

        preds_before = model.predict(self.fixture)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save_to_json(tmp_path)
            reloaded_model = SavedMLPPipeline.load_from_json(tmp_path)
            preds_after = reloaded_model.predict(self.fixture)

            self.assertEqual(preds_before, preds_after)
            self.assertEqual(reloaded_model.weights, weights)
            self.assertEqual(reloaded_model.bias, -0.2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_schema_version_mismatch_rejection(self):
        """Files with incompatible schema versions must fail clearly."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({
                "schema_version": "9.9.9",
                "input_shape": [4, 4],
                "weights": [0.0]*16,
                "bias": 0.0
            }, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                SavedMLPPipeline.load_from_json(tmp_path)
            self.assertIn("Schema/version mismatch", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_inference_input_schema_validation(self):
        """Malformed input during inference must raise ValueError."""
        model = SavedMLPPipeline()
        bad_records = [{"record_id": "r1", "pixels": [[0, 0], [0, 0]]}]
        with self.assertRaises(ValueError) as ctx:
            model.predict(bad_records)
        self.assertIn("4x4 matrix", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
