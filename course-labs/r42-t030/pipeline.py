"""
R42-T030: Save and reload small image pipeline config/parameters via JSON (no untrusted pickle).
Exposes prediction function with schema validation.
"""
from typing import Any, Dict, List
import json

class SavedPipeline:
    def __init__(self, threshold: int = 100, version: str = "1.0.0"):
        self.threshold = threshold
        self.version = version

    def save_config(self, filepath: str) -> None:
        data = {
            "version": self.version,
            "threshold": self.threshold
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_config(cls, filepath: str) -> "SavedPipeline":
        with open(filepath, "r") as f:
            data = json.load(f)
        if "threshold" not in data or "version" not in data:
            raise ValueError("Invalid pipeline config file")
        return cls(threshold=data["threshold"], version=data["version"])

    def predict(self, record: Dict[str, Any]) -> int:
        if not isinstance(record, dict) or "pixels" not in record:
            raise ValueError("Input record must contain pixels")
        if len(record["pixels"]) != 16:
            raise ValueError("Input pixels must have length 16")
        return 1 if sum(record["pixels"]) > self.threshold else 0
