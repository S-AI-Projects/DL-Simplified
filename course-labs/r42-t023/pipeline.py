"""
R42-T023: Minimal majority-class baseline for image classification.
Fits on training split only; evaluates on held-out test split.
"""
from typing import Any, Dict, List
from collections import Counter

class MajorityClassBaseline:
    def __init__(self):
        self.majority_label = None

    def fit(self, train_records: List[Dict[str, Any]]) -> None:
        if not train_records:
            raise ValueError("Training records cannot be empty")
        labels = [r["label"] for r in train_records]
        counts = Counter(labels)
        self.majority_label = counts.most_common(1)[0][0]

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.majority_label is None:
            raise RuntimeError("Model must be fitted before predict")
        preds = []
        for r in records:
            preds.append({
                "record_id": r["record_id"],
                "prediction": self.majority_label,
                "actual": r.get("label")
            })
        return preds

    def evaluate(self, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        preds = self.predict(test_records)
        correct = sum(1 for p in preds if p["prediction"] == p["actual"])
        total = len(preds)
        acc = correct / total if total > 0 else 0.0
        return {
            "total_test": total,
            "correct": correct,
            "accuracy": acc,
            "majority_label": self.majority_label
        }
