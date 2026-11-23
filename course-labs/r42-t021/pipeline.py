"""
R42-T021: Schema validator for DCGAN – MNIST digits image input records.
Fields: record_id (str), group_id (str), pixels (list of 16 ints [0, 255]), label (int 0 or 1).
Reports record ID, field and reason for each failure without silently dropping records.
"""
from typing import Any, Dict, List, Tuple

def validate_record(record: Any, seen_ids: set) -> List[Tuple[str, str, str]]:
    errors = []
    if not isinstance(record, dict):
        return [("unknown", "record", "Record must be a dictionary")]
    rec_id = record.get("record_id")
    if rec_id is None:
        errors.append(("unknown", "record_id", "Missing required field: record_id"))
        display_id = "unknown"
    elif not isinstance(rec_id, str) or not rec_id.strip():
        errors.append((str(rec_id), "record_id", "record_id must be a non-empty string"))
        display_id = str(rec_id)
    elif rec_id in seen_ids:
        errors.append((rec_id, "record_id", f"Duplicate record_id: {rec_id}"))
        display_id = rec_id
    else:
        seen_ids.add(rec_id)
        display_id = rec_id

    grp_id = record.get("group_id")
    if grp_id is None:
        errors.append((display_id, "group_id", "Missing required field: group_id"))
    elif not isinstance(grp_id, str) or not grp_id.strip():
        errors.append((display_id, "group_id", "group_id must be a non-empty string"))

    pixels = record.get("pixels")
    if pixels is None:
        errors.append((display_id, "pixels", "Missing required field: pixels"))
    elif not isinstance(pixels, list):
        errors.append((display_id, "pixels", "pixels must be a list of integers"))
    elif len(pixels) != 16:
        errors.append((display_id, "pixels", f"pixels length must be 16 (4x4), got {len(pixels)}"))
    else:
        for idx, val in enumerate(pixels):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0 or val > 255:
                errors.append((display_id, "pixels", f"pixel at index {idx} must be int in range [0, 255], got {val}"))
                break

    label = record.get("label")
    if label is None:
        errors.append((display_id, "label", "Missing required field: label"))
    elif not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
        errors.append((display_id, "label", f"label must be integer 0 or 1, got {label}"))

    return errors

def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen_ids = set()
    all_errors = []
    for r in records:
        errs = validate_record(r, seen_ids)
        for rec_id, field, reason in errs:
            all_errors.append({"record_id": rec_id, "field": field, "reason": reason})
    return {
        "total_records": len(records),
        "valid": len(all_errors) == 0,
        "error_count": len(all_errors),
        "errors": all_errors
    }
