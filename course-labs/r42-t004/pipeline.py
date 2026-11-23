"""
R42-T004 · Horses vs humans – CNN
Task type : Metric correctness (binary classification metrics)
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            plus fixed, simulated predictions per record (predictions.json).
NOTE: This is a CPU smoke fixture, NOT the Horses vs humans dataset. Metric
      values are pipeline verification, not real-world performance.

Metric definitions (labels are 0/1; positive class = 1)
-------------------------------------------------------
For a class c:  TP = truth==c and pred==c,  FP = truth!=c and pred==c,
                FN = truth==c and pred!=c.
  precision_c = TP / (TP + FP)
  recall_c    = TP / (TP + FN)
  f1_c        = 2·TP / (2·TP + FP + FN)
  accuracy    = (number of truth==pred) / N

Averaging
  "binary" : the metric for the positive class 1 only.
  "macro"  : unweighted mean of the per-class metric over classes 0 and 1
             (each class counts equally, regardless of its support).

Zero-denominator behaviour
  If a denominator is 0 (e.g. no predicted positives for precision), the
  metric for that class is `zero_division`: 0.0 (default) or 1.0, or
  zero_division="raise" raises ZeroDivisionError. With "macro", this
  replacement happens per class before averaging.
  Accuracy on zero records is undefined and raises ValueError.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from typing import Dict, List, Optional, Tuple, Union

LABELS = (0, 1)
POSITIVE = 1
ZeroDivision = Union[float, str]

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T004"


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _check(truth: List[int], prediction: List[int]) -> None:
    if len(truth) != len(prediction):
        raise ValueError(f"length mismatch: {len(truth)} truth vs {len(prediction)} predictions")
    if not truth:
        raise ValueError("metrics are undefined for zero records")
    for name, values in (("truth", truth), ("prediction", prediction)):
        for v in values:
            if isinstance(v, bool) or v not in LABELS:
                raise ValueError(f"{name} value {v!r} is not a label in {LABELS}")


def _check_zero_division(zero_division: ZeroDivision) -> None:
    if zero_division not in (0.0, 1.0, "raise") or isinstance(zero_division, bool):
        raise ValueError("zero_division must be 0.0, 1.0 or 'raise'")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def counts(truth: List[int], prediction: List[int], cls: int = POSITIVE) -> Tuple[int, int, int]:
    """Return (TP, FP, FN) for class `cls`."""
    _check(truth, prediction)
    tp = sum(1 for t, p in zip(truth, prediction) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(truth, prediction) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(truth, prediction) if t == cls and p != cls)
    return tp, fp, fn


def _divide(num: int, den: int, zero_division: ZeroDivision, what: str) -> float:
    if den == 0:
        if zero_division == "raise":
            raise ZeroDivisionError(f"{what}: denominator is 0")
        return float(zero_division)
    return num / den


def _per_class(kind: str, tp: int, fp: int, fn: int, zero_division: ZeroDivision) -> float:
    if kind == "precision":
        return _divide(tp, tp + fp, zero_division, "precision (TP+FP)")
    if kind == "recall":
        return _divide(tp, tp + fn, zero_division, "recall (TP+FN)")
    return _divide(2 * tp, 2 * tp + fp + fn, zero_division, "f1 (2TP+FP+FN)")


def _metric(kind: str, truth, prediction, average: str, zero_division: ZeroDivision) -> float:
    _check(truth, prediction)
    _check_zero_division(zero_division)
    if average == "binary":
        return _per_class(kind, *counts(truth, prediction, POSITIVE), zero_division)
    if average == "macro":
        values = [_per_class(kind, *counts(truth, prediction, c), zero_division) for c in LABELS]
        return sum(values) / len(values)
    raise ValueError(f"average must be 'binary' or 'macro', got {average!r}")


def precision(truth, prediction, average: str = "binary", zero_division: ZeroDivision = 0.0) -> float:
    return _metric("precision", truth, prediction, average, zero_division)


def recall(truth, prediction, average: str = "binary", zero_division: ZeroDivision = 0.0) -> float:
    return _metric("recall", truth, prediction, average, zero_division)


def f1(truth, prediction, average: str = "binary", zero_division: ZeroDivision = 0.0) -> float:
    return _metric("f1", truth, prediction, average, zero_division)


def accuracy(truth: List[int], prediction: List[int]) -> float:
    _check(truth, prediction)
    return sum(t == p for t, p in zip(truth, prediction)) / len(truth)


def evaluate(truth: List[int], prediction: List[int]) -> Dict[str, float]:
    """All metrics with the default zero_division=0.0."""
    report = {"accuracy": accuracy(truth, prediction)}
    for average in ("binary", "macro"):
        report[f"precision_{average}"] = precision(truth, prediction, average)
        report[f"recall_{average}"] = recall(truth, prediction, average)
        report[f"f1_{average}"] = f1(truth, prediction, average)
    return report


# ---------------------------------------------------------------------------
# Fixture evaluation
# ---------------------------------------------------------------------------

def load_fixture(path: Optional[str] = None) -> List[dict]:
    p = pathlib.Path(path) if path else LAB_DIR / "fixture.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_predictions(path: Optional[str] = None) -> Dict[str, int]:
    p = pathlib.Path(path) if path else LAB_DIR / "predictions.json"
    return json.loads(p.read_text(encoding="utf-8"))["predictions"]


def evaluate_fixture(records: List[dict], predictions: Dict[str, int]) -> dict:
    """
    Evaluate fixed predictions on the fixture (all six records form the test
    set; no model is trained, so there is no train/validation data here).
    Every record must have exactly one prediction.
    """
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id in fixture")
    missing = sorted(set(ids) - set(predictions))
    extra = sorted(set(predictions) - set(ids))
    if missing or extra:
        raise ValueError(f"prediction/record mismatch: missing={missing} extra={extra}")
    ordered = sorted(records, key=lambda r: r["record_id"])
    truth = [r["label"] for r in ordered]
    pred = [predictions[r["record_id"]] for r in ordered]
    tp, fp, fn = counts(truth, pred)
    return {
        "record_ids": [r["record_id"] for r in ordered],
        "truth": truth,
        "prediction": pred,
        "positive_class_counts": {"tp": tp, "fp": fp, "fn": fn},
        "metrics": evaluate(truth, pred),
    }


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, preds = load_fixture(), load_predictions()
    actual = {
        "task_example": {
            "truth": [1, 1, 0, 0],
            "prediction": [1, 0, 1, 0],
            "precision_binary": precision([1, 1, 0, 0], [1, 0, 1, 0]),
            "recall_binary": recall([1, 1, 0, 0], [1, 0, 1, 0]),
        },
        "fixture": evaluate_fixture(fixture, preds),
    }
    expected = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "predictions": preds}, indent=2), encoding="utf-8"
    )
    result = {
        "note": (
            "Pipeline verification only. Synthetic 4×4 fixture with simulated "
            "predictions — NOT the Horses vs humans dataset."
        ),
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "metric_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "third_party_packages": "none (standard library only)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[R42-T004] example precision={actual['task_example']['precision_binary']} "
          f"recall={actual['task_example']['recall_binary']} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
