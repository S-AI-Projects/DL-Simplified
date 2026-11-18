"""Minimal honest baseline for R42-T033."""
from typing import Any, Dict, List

class MajorityClassBaseline:
    def __init__(self):
        self.majority_label = None

    def fit(self, train_records: List[Dict[str, Any]]):
        if not train_records:
            raise ValueError("Training records cannot be empty")
        counts = {}
        for r in train_records:
            label = r["label"]
            counts[label] = counts.get(label, 0) + 1
        self.majority_label = max(counts.items(), key=lambda x: x[1])[0]

    def predict(self, test_records: List[Dict[str, Any]]) -> List[int]:
        if self.majority_label is None:
            raise RuntimeError("Model must be fit before calling predict")
        return [self.majority_label for _ in test_records]

    def evaluate(self, test_records: List[Dict[str, Any]]) -> Dict[str, float]:
        preds = self.predict(test_records)
        correct = sum(1 for p, r in zip(preds, test_records) if p == r["label"])
        acc = correct / len(test_records) if test_records else 0.0
        return {"accuracy": acc, "total_tested": len(test_records)}
