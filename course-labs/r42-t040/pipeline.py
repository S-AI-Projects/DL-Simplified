"""Safe saved-pipeline interface for R42-T040."""
import json
from typing import Any, Dict, List

class ColourisationPipeline:
    def __init__(self, scale_factor: float = 0.5):
        self.scale_factor = scale_factor
        self.version = "1.0"

    def predict(self, records: List[Dict[str, Any]]) -> List[List[List[float]]]:
        outputs = []
        for r in records:
            pixels = r.get("pixels")
            if not isinstance(pixels, list) or len(pixels) != 4:
                raise ValueError("Input pixels must be a 4x4 matrix")
            out_img = [[round(val * self.scale_factor, 4) for val in row] for row in pixels]
            outputs.append(out_img)
        return outputs

    def save_state(self, filepath: str):
        state = {
            "version": self.version,
            "scale_factor": self.scale_factor
        }
        with open(filepath, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, filepath: str) -> "ColourisationPipeline":
        with open(filepath, "r") as f:
            state = json.load(f)
        if state.get("version") != "1.0":
            raise ValueError(f"Unsupported pipeline state version: {state.get('version')}")
        return cls(scale_factor=state["scale_factor"])
