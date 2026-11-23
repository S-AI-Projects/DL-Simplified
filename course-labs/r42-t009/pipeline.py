"""
R42-T009 · FER-2013 facial expressions – CNN
Task type : Robustness fixture (one stated property, one unsupported input)
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture, NOT FER-2013. This is pipeline
      verification of ONE narrow property; it does NOT claim broad or
      adversarial robustness.

Stated property
---------------
Record order: shuffling or reversing the input records (IDs stay attached to
their records) does not change the result once predictions are aligned by
record_id. Images are not a time series, so reordering is a valid
transformation here.

Unsupported input
-----------------
Shape mismatch: a record whose pixels are not a flat list of 16 values (e.g.
15 values, or a nested 4×4 list) or that has no pixels field is REJECTED with
ValueError. There is no silent fallback (no padding, cropping or reshaping).

Pipeline: split by group (train g1 / validation g2 / test g3), fit a
nearest-centroid classifier on train (pixels / 255, squared Euclidean,
tie → smallest label), predict validation + test, return {record_id: label}.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
from typing import Dict, List, Optional

SPLITS = ("train", "validation", "test")
NUM_PIXELS = 16
PERMUTATION_SEED = 7

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T009"


def load_json(name: str):
    return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return load_json("fixture.json")


def check_input(rec: dict) -> List[float]:
    """Reject unsupported image input; return scaled features."""
    rid = rec.get("record_id")
    if "pixels" not in rec:
        raise ValueError(f"record {rid!r}: unsupported input, missing pixels field")
    pixels = rec["pixels"]
    if not isinstance(pixels, list) or any(isinstance(p, list) for p in pixels) or len(pixels) != NUM_PIXELS:
        got = len(pixels) if isinstance(pixels, list) else type(pixels).__name__
        raise ValueError(
            f"record {rid!r}: unsupported input, shape mismatch: expected flat list of {NUM_PIXELS} pixels, got {got}"
        )
    return [p / 255.0 for p in pixels]


def predict_by_id(records: List[dict], split: Dict[str, List[str]]) -> Dict[str, int]:
    """Fit on train, predict validation + test; result keyed by record_id."""
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id")
    owner = {gid: name for name in SPLITS for gid in split[name]}
    train = [r for r in records if owner[r["group_id"]] == "train"]
    held_out = [r for r in records if owner[r["group_id"]] != "train"]

    centroids: Dict[int, List[float]] = {}
    for label in sorted({r["label"] for r in train}):
        vecs = [check_input(r) for r in train if r["label"] == label]
        centroids[label] = [sum(col) / len(vecs) for col in zip(*vecs)]

    out: Dict[str, int] = {}
    for rec in held_out:
        x = check_input(rec)
        dists = {lab: sum((a - b) ** 2 for a, b in zip(x, c)) for lab, c in centroids.items()}
        best = min(dists.values())
        out[rec["record_id"]] = min(lab for lab, d in dists.items() if d == best)
    return out


def permute(records: List[dict], seed: int = PERMUTATION_SEED) -> List[dict]:
    """Seeded shuffle of record order; each record keeps its own record_id."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_json("split.json")
    shuffled = permute(fixture)
    bad = dict(fixture[5], pixels=fixture[5]["pixels"][:15])
    try:
        predict_by_id(fixture[:5] + [bad], split)
        rejection = "ACCEPTED (unexpected)"
    except ValueError as exc:
        rejection = f"rejected: {exc}"

    actual = {
        "original_order": [r["record_id"] for r in fixture],
        "original_predictions": predict_by_id(fixture, split),
        "shuffled_order": [r["record_id"] for r in shuffled],
        "shuffled_predictions_aligned": dict(sorted(predict_by_id(shuffled, split).items())),
        "unsupported_input": rejection,
    }
    actual["original_predictions"] = dict(sorted(actual["original_predictions"].items()))
    actual["property_holds"] = actual["original_predictions"] == actual["shuffled_predictions_aligned"]
    expected = load_json("expected_result.json")

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "split": split, "permutation_seed": PERMUTATION_SEED}, indent=2),
        encoding="utf-8",
    )
    result = {
        "note": ("Pipeline verification of one narrow property (record order). Synthetic 4×4 fixture — "
                 "NOT FER-2013. No claim of broad or adversarial robustness."),
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "robustness_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "third_party_packages": "none (standard library only)"}, indent=2),
        encoding="utf-8",
    )
    print(f"[R42-T009] property_holds={actual['property_holds']} {rejection} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
