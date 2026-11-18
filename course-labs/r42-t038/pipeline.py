"""Error analysis report generator for R42-T038."""
from typing import Any, Dict, List

def generate_error_report(records: List[Dict[str, Any]], predictions: Dict[str, int]) -> Dict[str, Any]:
    error_rows = []
    total_evaluated = 0

    for r in records:
        rid = r["record_id"]
        if rid not in predictions:
            continue
        total_evaluated += 1
        expected = r["label"]
        pred = predictions[rid]

        if pred != expected:
            category = "false_positive" if pred > expected else "false_negative"
            error_rows.append({
                "record_id": rid,
                "expected": expected,
                "predicted": pred,
                "reason_category": category
            })

    return {
        "total_evaluated": total_evaluated,
        "error_count": len(error_rows),
        "errors": error_rows
    }
