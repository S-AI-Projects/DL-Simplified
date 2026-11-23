"""
R42-T027: Serializable reproducible configuration.
Includes seed, schema version, feature ordering, split IDs, and algorithm settings.
Rejects unknown or contradictory settings.
"""
from typing import Any, Dict
import json

VALID_ALGORITHMS = {"skipgram", "cbow"}
ALLOWED_KEYS = {"schema_version", "seed", "algorithm", "embedding_dim", "window_size", "split_ids", "feature_order"}

def validate_and_serialize_config(config: Dict[str, Any]) -> str:
    # Check for unknown keys
    unknown = set(config.keys()) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration keys rejected: {unknown}")

    # Check required fields
    if "schema_version" not in config or not isinstance(config["schema_version"], str):
        raise ValueError("Missing or invalid schema_version")
    if "seed" not in config or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if config.get("algorithm") not in VALID_ALGORITHMS:
        raise ValueError(f"Contradictory or invalid algorithm: {config.get('algorithm')}")
    if config.get("embedding_dim", 0) <= 0:
        raise ValueError("embedding_dim must be positive")
    if config.get("window_size", 0) <= 0:
        raise ValueError("window_size must be positive")
    if not isinstance(config.get("split_ids", []), list) or not config["split_ids"]:
        raise ValueError("split_ids must be a non-empty list")

    return json.dumps(config, indent=2, sort_keys=True)

def load_config(json_str: str) -> Dict[str, Any]:
    cfg = json.loads(json_str)
    # validate on load
    validate_and_serialize_config(cfg)
    return cfg
