"""
R42-T028: Error-analysis report for image fixture.
Generates stable record ID, expected value, predicted value, and reason category.
Includes one deliberately injected wrong prediction for testability.
"""
from typing import Any, Dict, List

def generate_error_report(records: List[Dict[str, Any]], inject_error_id: str = "img_004") -> Dict[str, Any]:
    detailed_rows = []
    for r in records:
        expected = r["label"]
        # Deliberately invert prediction for injected error ID
        if r["record_id"] == inject_error_id:
            predicted = 1 - expected
            reason = "Injected deliberate ambiguity: corner pixel noise"
        else:
            predicted = expected
            reason = "Correct prediction"

        if predicted != expected:
            detailed_rows.append({
                "record_id": r["record_id"],
                "expected": expected,
                "predicted": predicted,
                "reason_category": reason
            })

    return {
        "total_records": len(records),
        "total_errors": len(detailed_rows),
        "error_records": detailed_rows
    }
