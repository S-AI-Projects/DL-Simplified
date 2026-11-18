"""Evaluation metric implementations and verification for R42-T034."""
from typing import Dict, List

def compute_classification_metrics(truth: List[int], pred: List[int]) -> Dict[str, float]:
    if len(truth) != len(pred):
        raise ValueError("Length mismatch between truth and predictions")
    if not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(truth)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def compute_mae(truth: List[float], pred: List[float]) -> float:
    if len(truth) != len(pred):
        raise ValueError("Length mismatch between truth and predictions")
    if not truth:
        return 0.0
    return sum(abs(t - p) for t, p in zip(truth, pred)) / len(truth)
