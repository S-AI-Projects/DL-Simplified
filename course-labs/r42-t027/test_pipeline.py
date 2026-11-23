import unittest
from pipeline import validate_and_serialize_config, load_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.valid_cfg = {
            "schema_version": "1.0.0",
            "seed": 42,
            "algorithm": "skipgram",
            "embedding_dim": 16,
            "window_size": 2,
            "split_ids": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "feature_order": ["text", "label"]
        }

    def test_valid_config_serialization(self):
        serialized = validate_and_serialize_config(self.valid_cfg)
        loaded = load_config(serialized)
        self.assertEqual(loaded["algorithm"], "skipgram")
        self.assertEqual(loaded["seed"], 42)

    def test_reject_unknown_setting(self):
        bad_cfg = self.valid_cfg.copy()
        bad_cfg["rogue_key"] = "dangerous"
        with self.assertRaises(ValueError):
            validate_and_serialize_config(bad_cfg)

    def test_reject_invalid_algorithm(self):
        bad_cfg = self.valid_cfg.copy()
        bad_cfg["algorithm"] = "unsupported_algo"
        with self.assertRaises(ValueError):
            validate_and_serialize_config(bad_cfg)

    def test_deliberate_failure(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.valid_cfg["seed"], 9999)

if __name__ == "__main__":
    unittest.main()
