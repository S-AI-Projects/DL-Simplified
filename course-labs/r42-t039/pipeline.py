"""Perturbation robustness interface test for R42-T039."""
from typing import Any, Dict, List

def compute_record_representations(records: List[Dict[str, Any]]) -> Dict[str, float]:
    results = {}
    for r in records:
        rid = r.get("record_id")
        pixels = r.get("pixels")
        if not rid or not isinstance(pixels, list) or len(pixels) != 4 or len(pixels[0]) != 4:
            raise ValueError(f"Malformed record {rid} or unsupported dimensions")
        avg_intensity = sum(sum(row) for row in pixels) / 16.0
        results[rid] = avg_intensity
    return results
