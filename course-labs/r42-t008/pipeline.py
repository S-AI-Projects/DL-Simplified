"""
R42-T008 · EuroSAT land use – CNN
Task type : Error-analysis example
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture, NOT EuroSAT. The report is pipeline
      verification only, not evidence of real-world performance.

Pipeline
--------
1. Split by group (split.json): train g1 / validation g2 / test g3.
2. Fit a nearest-centroid classifier (pixels / 255, squared Euclidean,
   tie → smallest label) on train only; predict validation + test.
3. Deliberately flip ONE prediction (default r006) so the report is testable.
4. Build the error report: one row per wrong prediction only, with
   record_id, split, expected, predicted, error_type and reason_category.
   Aggregate counts are computed from the rows and cross-checked.

Reason categories
  "injected_for_testing"   : the deliberately flipped prediction
  "nearer_to_wrong_centroid": a genuine model error
Error types
  "false_positive" (expected 0, predicted 1), "false_negative" (expected 1, predicted 0)
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from collections import Counter
from typing import Dict, List, Optional

SPLITS = ("train", "validation", "test")
EVALUATED = ("validation", "test")
INJECTED_RECORD_ID = "r006"

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T008"


def load_json(name: str):
    return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return load_json("fixture.json")


def split_records(records: List[dict], split: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id")
    owner: Dict[str, str] = {}
    for name in SPLITS:
        for gid in split[name]:
            if gid in owner:
                raise ValueError(f"group_id {gid!r} is in both {owner[gid]!r} and {name!r}")
            owner[gid] = name
    parts: Dict[str, List[dict]] = {name: [] for name in SPLITS}
    for rec in records:
        if rec["group_id"] not in owner:
            raise ValueError(f"record {rec['record_id']!r}: group {rec['group_id']!r} is in no split")
        parts[owner[rec["group_id"]]].append(rec)
    for name in SPLITS:
        parts[name].sort(key=lambda r: r["record_id"])
    return parts


# ---------------------------------------------------------------------------
# Model and predictions
# ---------------------------------------------------------------------------

def _features(rec: dict) -> List[float]:
    if len(rec["pixels"]) != 16:
        raise ValueError(f"record {rec['record_id']!r}: expected 16 pixels")
    return [p / 255.0 for p in rec["pixels"]]


def fit_centroids(train: List[dict]) -> Dict[int, List[float]]:
    centroids = {}
    for label in sorted({r["label"] for r in train}):
        vecs = [_features(r) for r in train if r["label"] == label]
        centroids[label] = [sum(col) / len(vecs) for col in zip(*vecs)]
    return centroids


def predict(centroids: Dict[int, List[float]], rec: dict) -> int:
    x = _features(rec)
    dists = {lab: sum((a - b) ** 2 for a, b in zip(x, c)) for lab, c in centroids.items()}
    best = min(dists.values())
    return min(lab for lab, d in dists.items() if d == best)


def make_predictions(records: List[dict], split: Dict[str, List[str]]) -> List[dict]:
    """Rows of {record_id, split, expected, predicted} for validation + test."""
    parts = split_records(records, split)
    centroids = fit_centroids(parts["train"])
    return [
        {"record_id": r["record_id"], "split": name,
         "expected": r["label"], "predicted": predict(centroids, r), "injected": False}
        for name in EVALUATED for r in parts[name]
    ]


def inject_wrong_prediction(rows: List[dict], record_id: str = INJECTED_RECORD_ID) -> List[dict]:
    """Return a copy of rows with the prediction for record_id deliberately flipped."""
    matches = [r for r in rows if r["record_id"] == record_id]
    if len(matches) != 1:
        raise ValueError(f"cannot inject: record_id {record_id!r} is not an evaluated record")
    out = []
    for r in rows:
        r = dict(r)
        if r["record_id"] == record_id:
            if r["predicted"] != r["expected"]:
                raise ValueError(f"cannot inject: prediction for {record_id!r} is already wrong")
            r["predicted"] = 1 - r["expected"]
            r["injected"] = True
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Error report
# ---------------------------------------------------------------------------

def error_report(rows: List[dict]) -> dict:
    """Detailed rows for wrong predictions only, plus aggregates that must match them."""
    errors = []
    for r in rows:
        if r["predicted"] == r["expected"]:
            continue
        errors.append({
            "record_id": r["record_id"],
            "split": r["split"],
            "expected": r["expected"],
            "predicted": r["predicted"],
            "error_type": "false_positive" if r["predicted"] == 1 else "false_negative",
            "reason_category": "injected_for_testing" if r["injected"] else "nearer_to_wrong_centroid",
        })
    errors.sort(key=lambda e: e["record_id"])
    aggregate = {
        "evaluated": len(rows),
        "correct": len(rows) - len(errors),
        "errors": len(errors),
        "by_split": {s: sum(e["split"] == s for e in errors) for s in EVALUATED},
        "by_error_type": dict(sorted(Counter(e["error_type"] for e in errors).items())),
        "by_reason_category": dict(sorted(Counter(e["reason_category"] for e in errors).items())),
    }
    check_aggregate(errors, aggregate)
    return {"rows": errors, "aggregate": aggregate}


def check_aggregate(errors: List[dict], aggregate: dict) -> None:
    """Raise if any aggregate count disagrees with the detailed rows."""
    n = len(errors)
    for key in ("by_split", "by_error_type", "by_reason_category"):
        if sum(aggregate[key].values()) != n:
            raise ValueError(f"aggregate {key} sums to {sum(aggregate[key].values())}, rows = {n}")
    if aggregate["errors"] != n or aggregate["correct"] + n != aggregate["evaluated"]:
        raise ValueError("aggregate totals do not match detailed rows")


def run_report(records: List[dict], split: Dict[str, List[str]],
               record_id: str = INJECTED_RECORD_ID) -> dict:
    return error_report(inject_wrong_prediction(make_predictions(records, split), record_id))


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_json("split.json")
    actual = run_report(fixture, split)
    expected = load_json("expected_report.json")

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "split": split, "injected_record_id": INJECTED_RECORD_ID}, indent=2),
        encoding="utf-8",
    )
    result = {
        "note": "Pipeline verification only. Synthetic 4×4 fixture — NOT EuroSAT.",
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "error_report_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "third_party_packages": "none (standard library only)"}, indent=2),
        encoding="utf-8",
    )
    print(f"[R42-T008] errors={actual['aggregate']['errors']} rows={actual['rows']} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
