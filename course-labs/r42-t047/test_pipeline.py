"""Unit tests for R42-T047 reproducible configuration."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    build_reproducible_config,
    deserialize_config,
    serialize_config,
    validate_config
)


class TestReproducibleConfig(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)
        self.split_ids = {
            "train": [r["record_id"] for r in self.fixture[:4]],
            "test": [r["record_id"] for r in self.fixture[4:]]
        }

    def test_reproducible_serialization_cycle(self):
        """Serialize and deserialize config, ensuring reconstructed config is identical."""
        cfg = build_reproducible_config(self.split_ids, seed=42)
        json_str = serialize_config(cfg)
        reloaded = deserialize_config(json_str)

        self.assertEqual(cfg, reloaded)
        self.assertEqual(reloaded["seed"], 42)
        self.assertEqual(reloaded["split_ids"]["train"], ["rec_001", "rec_002", "rec_003", "rec_004"])

    def test_reject_unknown_field(self):
        """Unknown top-level field must be rejected."""
        cfg = build_reproducible_config(self.split_ids)
        cfg["unexpected_hyperparameter"] = 123
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Unknown configuration key", str(ctx.exception))

    def test_reject_incompatible_input_shape(self):
        """Incompatible image input shape must be rejected."""
        cfg = build_reproducible_config(self.split_ids, input_shape=[28, 28])
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("Incompatible input shape", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
