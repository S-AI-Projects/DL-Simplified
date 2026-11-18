import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import PipelineConfig

class TestPipelineConfig(unittest.TestCase):
    def setUp(self):
        self.valid_cfg = {
            "schema_version": "1.0",
            "seed": 42,
            "feature_shape": [4, 4],
            "noise_level": 0.1,
            "split_ids": ["rec_001", "rec_002"],
            "algorithm": "denoising_autoencoder_cpu"
        }

    def test_valid_config_serialization(self):
        cfg = PipelineConfig(self.valid_cfg)
        serialized = cfg.to_json()
        reconstructed = PipelineConfig.from_json(serialized)
        self.assertEqual(cfg.data, reconstructed.data)

    def test_unknown_key_rejected(self):
        bad_cfg = dict(self.valid_cfg)
        bad_cfg["unsupported_field"] = "malicious"
        with self.assertRaises(ValueError):
            PipelineConfig(bad_cfg)

    def test_contradictory_noise_rejected(self):
        bad_cfg = dict(self.valid_cfg)
        bad_cfg["noise_level"] = -0.5
        with self.assertRaises(ValueError):
            PipelineConfig(bad_cfg)

    def test_incompatible_shape_rejected(self):
        bad_cfg = dict(self.valid_cfg)
        bad_cfg["feature_shape"] = [28, 28]
        with self.assertRaises(ValueError):
            PipelineConfig(bad_cfg)

if __name__ == "__main__":
    unittest.main()
