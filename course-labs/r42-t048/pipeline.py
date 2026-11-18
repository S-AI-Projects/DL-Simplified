"""Error analysis report module for Data augmentation study - small image set.

Generates a structured error analysis report containing:
- Stable record ID
- Expected value (truth)
- Predicted value
- Reason category (e.g. FALSE_POSITIVE, FALSE_NEGATIVE, NOISE_SENSITIVITY)
- Summary statistics (total_errors, error_rate)
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from typing import Any, Dict, List


def generate_error_report(
    records: List[Dict[str, Any]],
    predictions: List[int]
) -> Dict[str, Any]:
    """Generate structured error report for mismatched predictions."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions count must match")

    error_rows: List[Dict[str, Any]] = []
    correct_count = 0

    for rec, pred in zip(records, predictions):
        rec_id = rec["record_id"]
        truth = rec["label"]
        if truth != pred:
            if truth == 0 and pred == 1:
                category = "FALSE_POSITIVE_DIAGONAL_TRIGGER"
            elif truth == 1 and pred == 0:
                category = "FALSE_NEGATIVE_MISSED_PATTERN"
            else:
                category = "UNKNOWN_MISCLASSIFICATION"

            error_rows.append({
                "record_id": rec_id,
                "expected": truth,
                "predicted": pred,
                "reason_category": category
            })
        else:
            correct_count += 1

    total_records = len(records)
    total_errors = len(error_rows)
    error_rate = total_errors / total_records if total_records > 0 else 0.0

    return {
        "total_records": total_records,
        "correct_predictions": correct_count,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "error_records": error_rows
    }
