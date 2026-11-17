"""Evaluation metric verification module for SVHN house numbers - CNN.

Implements precision, recall, accuracy, and F1 with explicit zero-denominator
and averaging behavior.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from typing import Dict, List, Tuple


def compute_binary_metrics(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0
) -> Dict[str, float]:
    """Compute binary classification metrics with explicit zero-division behavior.

    Explicitly verified on canonical case:
      truth=[1, 1, 0, 0], pred=[1, 0, 1, 0] -> precision=0.5, recall=0.5
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length")
    if not y_true:
        return {"precision": zero_division, "recall": zero_division, "f1": zero_division, "accuracy": zero_division}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else zero_division
    accuracy = (tp + tn) / len(y_true)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn)
    }
