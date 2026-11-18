"""Saved pipeline interface module for MNIST - MLP from scratch in NumPy.

Saves and reloads small image pipeline parameters using JSON.
Exposes a schema-validated predict function. Never loads untrusted pickles.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

import json
import math
from typing import Any, Dict, List


MODEL_SCHEMA_VERSION = "1.0.0"


class SavedMLPPipeline:
    """Small MLP pipeline that can be serialized/deserialized via pure JSON."""

    def __init__(self, weights: List[float] = None, bias: float = 0.0):
        # 16 weights for 4x4 image
        self.weights = weights if weights is not None else [0.0] * 16
        self.bias = bias
        self.version = MODEL_SCHEMA_VERSION

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def save_to_json(self, filepath: str) -> None:
        """Serialize parameters to JSON file."""
        data = {
            "model_type": "TinyMLPBinaryClassifier",
            "schema_version": self.version,
            "input_shape": [4, 4],
            "weights": self.weights,
            "bias": self.bias
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "SavedMLPPipeline":
        """Deserialize parameters safely from JSON with schema validation."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Corrupt model file: expected dictionary")

        if data.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(f"Schema/version mismatch: expected {MODEL_SCHEMA_VERSION}, got {data.get('schema_version')}")

        if data.get("input_shape") != [4, 4]:
            raise ValueError(f"Incompatible model input shape: {data.get('input_shape')}")

        weights = data.get("weights")
        if not isinstance(weights, list) or len(weights) != 16:
            raise ValueError(f"Invalid weights array: expected 16 floats, got {len(weights) if isinstance(weights, list) else type(weights)}")

        bias = data.get("bias", 0.0)
        return cls(weights=weights, bias=float(bias))

    def predict(self, records: List[Dict[str, Any]], threshold: float = 0.5) -> List[int]:
        """Perform predictions on input records with schema validation."""
        predictions = []
        for r in records:
            if not isinstance(r, dict) or "pixels" not in r:
                raise ValueError("Record must be a dict containing 'pixels'")
            pixels = r["pixels"]
            if not isinstance(pixels, list) or len(pixels) != 4:
                raise ValueError("Input pixels must be a 4x4 matrix")

            flat = []
            for row in pixels:
                if not isinstance(row, list) or len(row) != 4:
                    raise ValueError("Each row in pixels must have 4 columns")
                for val in row:
                    flat.append(float(val) / 255.0)

            z = sum(w * x for w, x in zip(self.weights, flat)) + self.bias
            prob = self._sigmoid(z)
            predictions.append(1 if prob >= threshold else 0)

        return predictions
