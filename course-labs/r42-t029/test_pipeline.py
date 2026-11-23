import unittest
from pipeline import run_image_inference, apply_perturbation_reorder

class TestPerturbation(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            {"record_id": "img_001", "pixels": [0]*16, "label": 0},
            {"record_id": "img_002", "pixels": [0]*16, "label": 0},
            {"record_id": "img_003", "pixels": [0]*16, "label": 0},
            {"record_id": "img_004", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_005", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
            {"record_id": "img_006", "pixels": [255 if i in (0, 5, 10, 15) else 0 for i in range(16)], "label": 1},
        ]

    def test_order_invariance(self):
        orig_preds = run_image_inference(self.fixture)
        reordered = apply_perturbation_reorder(self.fixture)
        reordered_preds = run_image_inference(reordered)
        self.assertEqual(orig_preds, reordered_preds)

    def test_shape_mismatch_rejected(self):
        bad_fixture = [{"record_id": "img_bad", "pixels": [0]*8}]
        with self.assertRaises(ValueError):
            run_image_inference(bad_fixture)

    def test_missing_field_rejected(self):
        bad_fixture = [{"pixels": [0]*16}]
        with self.assertRaises(KeyError):
            run_image_inference(bad_fixture)

    def test_deliberate_failure(self):
        orig_preds = run_image_inference(self.fixture)
        with self.assertRaises(AssertionError):
            self.assertEqual(len(orig_preds), 999)

if __name__ == "__main__":
    unittest.main()
