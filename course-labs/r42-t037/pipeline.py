"""Reproducible configuration manager for R42-T037."""
import json
from typing import Any, Dict

ALLOWED_KEYS = {"schema_version", "seed", "feature_shape", "noise_level", "split_ids", "algorithm"}

class PipelineConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.validate(config_dict)
        self.data = dict(config_dict)

    @staticmethod
    def validate(d: Dict[str, Any]):
        for k in d.keys():
            if k not in ALLOWED_KEYS:
                raise ValueError(f"Unknown configuration key: {k}")
        
        if d.get("schema_version") != "1.0":
            raise ValueError("Unsupported schema_version; expected '1.0'")
            
        if not isinstance(d.get("seed"), int):
            raise ValueError("seed must be an integer")

        noise = d.get("noise_level", 0.0)
        if not isinstance(noise, (int, float)) or not (0.0 <= noise <= 1.0):
            raise ValueError("noise_level must be a float between 0.0 and 1.0")

        shape = d.get("feature_shape")
        if shape != [4, 4]:
            raise ValueError(f"Incompatible feature shape: {shape}, expected [4, 4]")

    def to_json(self) -> str:
        return json.dumps(self.data, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineConfig":
        d = json.loads(json_str)
        return cls(d)
