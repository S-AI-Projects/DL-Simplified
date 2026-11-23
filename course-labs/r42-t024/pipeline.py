"""
R42-T024: Cross-checked evaluation metrics (Precision, Recall, F1, Accuracy) with explicit zero-denominator behavior.
"""
from typing import Dict, List, Tuple

def compute_binary_metrics(actuals: List[int], predictions: List[int], zero_division: float = 0.0) -> Dict[str, float]:
    if len(actuals) != len(predictions):
        raise ValueError("Length mismatch between actuals and predictions")
    if not actuals:
        return {"accuracy": 0.0, "precision": zero_division, "recall": zero_division, "f1": zero_division}

    tp = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 0)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)

    total = len(actuals)
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else zero_division

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
