"""Pipeline schema validator for Hand gesture images - CNN.

Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from typing import Any, Dict, List, Tuple


REQUIRED_FIELDS = ("record_id", "group_id", "pixels", "label")


def validate_record(record: Any, seen_ids: set) -> List[Tuple[str, str, str]]:
    """Validate a single image record.

    Returns a list of failures formatted as (record_id, field, reason).
    """
    failures: List[Tuple[str, str, str]] = []

    if not isinstance(record, dict):
        return [("UNKNOWN", "record", "Record must be a dictionary / JSON object")]

    raw_id = record.get("record_id")
    if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
        rec_id = "UNKNOWN"
        failures.append((rec_id, "record_id", "Missing or empty string record_id"))
    else:
        rec_id = raw_id.strip()
        if rec_id in seen_ids:
            failures.append((rec_id, "record_id", f"Duplicate record ID '{rec_id}' encountered"))
        seen_ids.add(rec_id)

    raw_group = record.get("group_id")
    if raw_group is None or not isinstance(raw_group, str) or not raw_group.strip():
        failures.append((rec_id, "group_id", "Missing or non-string group_id"))

    raw_label = record.get("label")
    if raw_label is None:
        failures.append((rec_id, "label", "Missing label field"))
    elif not isinstance(raw_label, int) or raw_label not in (0, 1):
        failures.append((rec_id, "label", f"Label must be integer 0 or 1, got {raw_label}"))

    pixels = record.get("pixels")
    if pixels is None:
        failures.append((rec_id, "pixels", "Missing pixels field"))
    elif not isinstance(pixels, list):
        failures.append((rec_id, "pixels", "Pixels must be a 2D list of shape 4x4"))
    else:
        if len(pixels) != 4:
            failures.append((rec_id, "pixels", f"Pixels height must be 4, got {len(pixels)}"))
        else:
            for r_idx, row in enumerate(pixels):
                if not isinstance(row, list) or len(row) != 4:
                    failures.append((rec_id, "pixels", f"Row {r_idx} must have 4 columns"))
                    break
                for c_idx, val in enumerate(row):
                    if not isinstance(val, (int, float)) or not (0 <= val <= 255):
                        failures.append((rec_id, "pixels", f"Pixel value at ({r_idx}, {c_idx}) must be in range [0, 255]"))
                        break

    return failures


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate full dataset of records without silently dropping any record.

    Returns:
        Dict with 'is_valid': bool, 'total_records': int, 'failures': list of (rec_id, field, reason).
    """
    seen_ids = set()
    all_failures: List[Tuple[str, str, str]] = []

    if not isinstance(records, list):
        return {
            "is_valid": False,
            "total_records": 0,
            "failures": [("GLOBAL", "dataset", "Dataset must be a list of records")]
        }

    for record in records:
        rec_failures = validate_record(record, seen_ids)
        all_failures.extend(rec_failures)

    return {
        "is_valid": len(all_failures) == 0,
        "total_records": len(records),
        "failures": all_failures
    }
