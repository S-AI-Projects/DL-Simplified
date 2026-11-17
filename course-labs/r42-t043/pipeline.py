"""Minimal baseline classifier for Oxford-IIIT pets - transfer learning fixture.

Implements majority-class classifier trained strictly on training records.
Calculates accuracy and macro F1 on a held-out test split.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple


class MajorityClassBaseline:
    """Majority-class baseline classifier."""

    def __init__(self):
        self.majority_label = None

    def fit(self, train_records: List[Dict[str, Any]]) -> "MajorityClassBaseline":
        if not train_records:
            raise ValueError("Training records cannot be empty")
        labels = [r["label"] for r in train_records]
        counts = Counter(labels)
        # In case of tie, select min label deterministically
        most_common = counts.most_common()
        top_count = most_common[0][1]
        candidates = [lbl for lbl, cnt in most_common if cnt == top_count]
        self.majority_label = sorted(candidates)[0]
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        if self.majority_label is None:
            raise RuntimeError("Model must be fit before calling predict")
        return [self.majority_label] * len(records)


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate accuracy and macro F1 score."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / len(y_true)

    classes = sorted(list(set(y_true) | set(y_pred)))
    f1_list = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}
