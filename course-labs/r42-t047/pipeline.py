"""Reproducible configuration serialization and verification module.

Project/section: Yoga pose images – CNN.
Writes a serializable configuration for the image fixture with seed, schema version,
feature ordering, split IDs, and algorithm settings; rejects unknown or contradictory settings.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

import json
from typing import Any, Dict, List


ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "seed",
    "feature_ordering",
    "split_ids",
    "algorithm_settings",
    "environment"
}

ALLOWED_ALGORITHM_KEYS = {
    "learning_rate",
    "max_steps",
    "input_shape"
}


def build_reproducible_config(
    split_ids: Dict[str, List[str]],
    seed: int = 42,
    schema_version: str = "1.0.0",
    feature_ordering: str = "row_major",
    learning_rate: float = 0.05,
    max_steps: int = 10,
    input_shape: List[int] = None
) -> Dict[str, Any]:
    """Generate serializable config dictionary."""
    if input_shape is None:
        input_shape = [4, 4, 1]

    return {
        "schema_version": schema_version,
        "seed": seed,
        "feature_ordering": feature_ordering,
        "split_ids": split_ids,
        "algorithm_settings": {
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "input_shape": input_shape
        },
        "environment": {
            "framework": "pure_python_stdlib",
            "device": "cpu"
        }
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration schema, checking for unknown or contradictory settings."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    # Reject unknown keys
    unknown_keys = set(config.keys()) - ALLOWED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown configuration key(s) detected: {unknown_keys}")

    # Validate seed
    if not isinstance(config.get("seed"), int):
        raise ValueError("Config seed must be an integer")

    # Validate schema version
    if config.get("schema_version") not in ("1.0.0", "1.1.0"):
        raise ValueError(f"Unsupported schema version: {config.get('schema_version')}")

    # Validate algorithm settings
    algo = config.get("algorithm_settings")
    if not isinstance(algo, dict):
        raise ValueError("algorithm_settings must be a dictionary")

    unknown_algo_keys = set(algo.keys()) - ALLOWED_ALGORITHM_KEYS
    if unknown_algo_keys:
        raise ValueError(f"Unknown algorithm setting(s) detected: {unknown_algo_keys}")

    # Check contradictory or incompatible settings
    input_shape = algo.get("input_shape")
    if input_shape != [4, 4, 1] and input_shape != [4, 4]:
        raise ValueError(f"Incompatible input shape {input_shape}; expected [4, 4, 1] or [4, 4]")

    if algo.get("learning_rate", 0) <= 0:
        raise ValueError("Learning rate must be strictly positive")
    if algo.get("max_steps", 0) <= 0:
        raise ValueError("max_steps must be strictly positive")

    # Check split IDs integrity
    splits = config.get("split_ids")
    if not isinstance(splits, dict) or "train" not in splits or "test" not in splits:
        raise ValueError("split_ids must define at least 'train' and 'test'")


def serialize_config(config: Dict[str, Any]) -> str:
    """Validate and serialize config to JSON string."""
    validate_config(config)
    return json.dumps(config, indent=2, sort_keys=True)


def deserialize_config(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string and validate config."""
    config = json.loads(json_str)
    validate_config(config)
    return config
