"""
R42-T003 · Grad-CAM explanations – any CNN
Task type : Baseline comparison (minimal honest baseline)
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture. It is NOT a real image dataset and the
      numbers below are pipeline verification, not real-world accuracy.

Baseline
--------
Majority-class classifier, the simplest honest baseline for an image
classification contract. Its single parameter (the majority label) is fitted
on the TRAIN split only; predictions and accuracy are computed on the fixed
held-out validation and test splits. Ties are broken towards the smallest
label. It looks at no pixels, so it produces no spatial (Grad-CAM-style)
attribution: it is the floor a CNN (and its explanations) must beat.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from collections import Counter
from typing import Dict, List, Optional

SPLITS = ("train", "validation", "test")
VALID_LABELS = (0, 1)

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T003"


# ---------------------------------------------------------------------------
# Fixture and split
# ---------------------------------------------------------------------------

def load_fixture(path: Optional[str] = None) -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    p = pathlib.Path(path) if path else LAB_DIR / "fixture.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_split(path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load the fixed group-level split: {split: [group_id, ...]}."""
    p = pathlib.Path(path) if path else LAB_DIR / "split.json"
    return json.loads(p.read_text(encoding="utf-8"))


def split_records(records: List[dict], split: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    """
    Assign records to train / validation / test by group_id.
    Rejects duplicate record IDs, groups listed in more than one split,
    and records whose group is in no split.
    """
    ids = [r["record_id"] for r in records]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate record_id(s): {dupes}")

    group_to_split: Dict[str, str] = {}
    for name in SPLITS:
        for gid in split[name]:
            if gid in group_to_split:
                raise ValueError(
                    f"group_id {gid!r} is in both {group_to_split[gid]!r} and {name!r}"
                )
            group_to_split[gid] = name

    out: Dict[str, List[dict]] = {name: [] for name in SPLITS}
    for rec in records:
        name = group_to_split.get(rec["group_id"])
        if name is None:
            raise ValueError(f"record {rec['record_id']!r}: group {rec['group_id']!r} is in no split")
        out[name].append(rec)
    for name in SPLITS:
        if not out[name]:
            raise ValueError(f"split {name!r} is empty")
    return out


# ---------------------------------------------------------------------------
# Majority-class baseline
# ---------------------------------------------------------------------------

class MajorityClassBaseline:
    """Predicts the most frequent training label for every input."""

    def __init__(self) -> None:
        self.majority_label: Optional[int] = None
        self.train_label_counts: Dict[int, int] = {}

    def fit(self, train_records: List[dict]) -> "MajorityClassBaseline":
        if not train_records:
            raise ValueError("cannot fit on an empty training split")
        labels = [r["label"] for r in train_records]
        bad = [l for l in labels if l not in VALID_LABELS]
        if bad:
            raise ValueError(f"invalid training label(s): {bad}")
        counts = Counter(labels)
        top = max(counts.values())
        self.majority_label = min(l for l, c in counts.items() if c == top)  # tie → smallest
        self.train_label_counts = {l: counts.get(l, 0) for l in VALID_LABELS}
        return self

    def predict(self, records: List[dict]) -> List[int]:
        if self.majority_label is None:
            raise RuntimeError("baseline is not fitted")
        return [self.majority_label for _ in records]


def accuracy(truth: List[int], prediction: List[int]) -> float:
    """Fraction of exact matches. Empty or length-mismatched inputs are rejected."""
    if len(truth) != len(prediction):
        raise ValueError(f"length mismatch: {len(truth)} truth vs {len(prediction)} predictions")
    if not truth:
        raise ValueError("accuracy is undefined for zero records")
    return sum(t == p for t, p in zip(truth, prediction)) / len(truth)


def run_baseline(records: List[dict], split: Dict[str, List[str]]) -> dict:
    """Fit on train only; report predictions and accuracy on validation and test."""
    parts = split_records(records, split)
    model = MajorityClassBaseline().fit(parts["train"])
    result = {
        "baseline": "majority-class",
        "fitted_on": sorted(r["record_id"] for r in parts["train"]),
        "train_label_counts": {str(k): v for k, v in model.train_label_counts.items()},
        "majority_label": model.majority_label,
        "metric": "accuracy",
    }
    for name in ("validation", "test"):
        recs = sorted(parts[name], key=lambda r: r["record_id"])
        preds = model.predict(recs)
        result[name] = {
            "record_ids": [r["record_id"] for r in recs],
            "truth": [r["label"] for r in recs],
            "prediction": preds,
            "accuracy": accuracy([r["label"] for r in recs], preds),
        }
    return result


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    """Run the baseline and save inputs, expected vs actual, and tool versions."""
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_split()
    actual = run_baseline(fixture, split)
    expected = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "split": split}, indent=2), encoding="utf-8"
    )
    result = {
        "note": (
            "Pipeline verification only. Synthetic 4×4 fixture — not a real "
            "dataset; no real-world accuracy claim."
        ),
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "baseline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    print(f"[R42-T003] majority label={actual['majority_label']} "
          f"val acc={actual['validation']['accuracy']} test acc={actual['test']['accuracy']} "
          f"outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
