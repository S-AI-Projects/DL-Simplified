"""
R42-T029: Robustness property testing on image interface.
Tests record order change with stable IDs, missing fields, and shape mismatches.
"""
from typing import Any, Dict, List
import copy

def run_image_inference(records: List[Dict[str, Any]]) -> Dict[str, int]:
    results = {}
    for r in records:
        if "record_id" not in r:
            raise KeyError("Missing required field: record_id")
        if "pixels" not in r or len(r["pixels"]) != 16:
            raise ValueError(f"Invalid or missing pixels in record {r.get('record_id')}")
        # Deterministic dummy prediction based on pixel sum
        pred = 1 if sum(r["pixels"]) > 100 else 0
        results[r["record_id"]] = pred
    return results

def apply_perturbation_reorder(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Valid perturbation: reverse record order
    return list(reversed(records))
