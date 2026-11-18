"""Data contract and schema validator for R42-T031 Kuzushiji-MNIST - CNN."""
from typing import Any, Dict, List, Set, Tuple

def validate_record(record: Dict[str, Any], seen_ids: Set[str]) -> List[Tuple[str, str, str]]:
    failures = []
    rec_id = record.get("record_id")
    if not rec_id or not isinstance(rec_id, str):
        failures.append((str(rec_id), "record_id", "record_id must be a non-empty string"))
    elif rec_id in seen_ids:
        failures.append((rec_id, "record_id", f"Duplicate record ID: {rec_id}"))

    group_id = record.get("group_id")
    if not group_id or not isinstance(group_id, str):
        failures.append((str(rec_id), "group_id", "group_id must be a non-empty string"))

    pixels = record.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != 4:
        failures.append((str(rec_id), "pixels", "pixels must be a 4x4 list of lists"))
    else:
        valid_shape = True
        for row in pixels:
            if not isinstance(row, list) or len(row) != 4:
                failures.append((str(rec_id), "pixels", "each row in pixels must have length 4"))
                valid_shape = False
                break
            for val in row:
                if not isinstance(val, (int, float)) or not (0 <= val <= 255):
                    failures.append((str(rec_id), "pixels", f"pixel values must be numeric in [0, 255], got {val}"))
                    valid_shape = False
                    break
            if not valid_shape:
                break

    label = record.get("label")
    if label not in (0, 1):
        failures.append((str(rec_id), "label", f"label must be binary (0 or 1), got {label}"))

    return failures

def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_failures = []
    seen_ids = set()
    for rec in records:
        f = validate_record(rec, seen_ids)
        if f:
            all_failures.extend(f)
        if isinstance(rec.get("record_id"), str) and rec.get("record_id"):
            seen_ids.add(rec["record_id"])
    return {
        "total_records": len(records),
        "is_valid": len(all_failures) == 0,
        "failures": all_failures
    }
