"""Explicit preprocessing pipeline for Garbage classification - transfer learning.

Scales pixels from [0..255] to [0.0..1.0] with an explicit channel axis: shape (4, 4, 1).
Rejects invalid dimensions, negative values, and out-of-range pixels (> 255).
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from typing import Any, Dict, List, Tuple


def preprocess_image(pixels: Any) -> List[List[List[float]]]:
    """Convert 4x4 2D list of pixel values into normalized 4x4x1 3D list (float32 emulation).

    Raises:
        ValueError if dimensions are invalid or pixel values are out of [0, 255].
    """
    if not isinstance(pixels, list) or len(pixels) != 4:
        raise ValueError(f"Expected 4 rows, got {len(pixels) if isinstance(pixels, list) else type(pixels)}")

    tensor_4x4x1: List[List[List[float]]] = []
    for r_idx, row in enumerate(pixels):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"Row {r_idx} must have 4 columns")
        out_row: List[List[float]] = []
        for c_idx, val in enumerate(row):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Pixel at ({r_idx}, {c_idx}) must be numeric, got {type(val)}")
            if val < 0.0 or val > 255.0:
                raise ValueError(f"Pixel at ({r_idx}, {c_idx}) out of range [0, 255]: {val}")
            scaled = float(val) / 255.0
            out_row.append([scaled])
        tensor_4x4x1.append(out_row)

    return tensor_4x4x1


def preprocess_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform records by adding preprocessed 'tensor' field with shape (4, 4, 1)."""
    processed = []
    for r in records:
        tensor = preprocess_image(r["pixels"])
        processed.append({
            "record_id": r["record_id"],
            "group_id": r["group_id"],
            "label": r["label"],
            "tensor": tensor
        })
    return processed
