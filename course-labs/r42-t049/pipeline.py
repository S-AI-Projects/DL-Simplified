"""Input perturbation robustness test for Transfer learning comparison - MobileNet vs ResNet.

Tests robustness properties:
1. Valid perturbation: Record order permutation with stable record ID tracking.
   Predictions remain invariant when aligned by record ID.
2. Invalid perturbations: Missing required field or shape mismatch.
   Rejected with clear errors or mapped to explicit fallbacks.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from typing import Any, Dict, List


def classify_record(record: Dict[str, Any]) -> int:
    """Predict label based on image content (diagonal intensity sum vs 0).

    Raises ValueError if record format or shape is invalid.
    """
    if not isinstance(record, dict):
        raise ValueError("Record must be a dictionary")
    if "record_id" not in record or "pixels" not in record:
        raise ValueError("Record is missing required 'record_id' or 'pixels' field")

    pixels = record["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 4:
        raise ValueError("Invalid pixel dimensions; expected 4 rows")

    diag_sum = 0
    for i in range(4):
        if not isinstance(pixels[i], list) or len(pixels[i]) != 4:
            raise ValueError(f"Invalid columns in row {i}; expected 4 columns")
        diag_sum += pixels[i][i]

    # If diagonal contains white intensity (> 0), predict 1, else 0
    return 1 if diag_sum > 0 else 0


def run_pipeline_with_order_invariance(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Run classification and return predictions indexed by record_id."""
    return {r["record_id"]: classify_record(r) for r in records}
