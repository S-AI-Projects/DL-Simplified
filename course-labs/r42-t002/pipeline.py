"""
R42-T002 · NEU steel surface defects – CNN
Task type : Leakage-free splitting (deterministic split manifest)
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture. It is NOT the NEU steel surface defects
      dataset and must not be used to make performance claims about it.

Split rules
-----------
* Splitting is done per group: all records of one group_id go to exactly one
  of train / validation / test, so no group leaks across splits.
* Without timestamps, groups are sorted and then shuffled with a fixed seed
  (random.Random(seed)), so the same input + seed always gives the same manifest.
* If every record has a "timestamp", groups are ordered by their earliest
  timestamp instead (oldest → train, newest → test) and the split must be
  chronological: every train time <= every validation time <= every test time.
* Duplicate record IDs, missing IDs/groups, and manifests whose splits overlap
  by record ID or group are rejected with ValueError.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
from typing import Dict, List, Optional

SPLITS = ("train", "validation", "test")
DEFAULT_SEED = 42
DEFAULT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T002"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def load_fixture(path: Optional[str] = None) -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    p = pathlib.Path(path) if path else LAB_DIR / "fixture.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def check_records(records: List[dict]) -> None:
    """Reject records that would make a split ambiguous or leaky."""
    seen = set()
    for i, rec in enumerate(records):
        rid = rec.get("record_id") if isinstance(rec, dict) else None
        gid = rec.get("group_id") if isinstance(rec, dict) else None
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"record {i}: missing or invalid record_id")
        if not isinstance(gid, str) or not gid:
            raise ValueError(f"record {rid!r}: missing or invalid group_id")
        if rid in seen:
            raise ValueError(f"duplicate record_id {rid!r}")
        seen.add(rid)

    has_ts = ["timestamp" in r for r in records]
    if any(has_ts) and not all(has_ts):
        raise ValueError("timestamp must be present on all records or on none")


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------

def _group_counts(n_groups: int, fractions: Dict[str, float]) -> Dict[str, int]:
    if n_groups < len(SPLITS):
        raise ValueError(
            f"need at least {len(SPLITS)} groups for train/validation/test, got {n_groups}"
        )
    n_val = max(1, round(n_groups * fractions["validation"]))
    n_test = max(1, round(n_groups * fractions["test"]))
    n_train = n_groups - n_val - n_test
    if n_train < 1:
        raise ValueError("fractions leave no group for train")
    return {"train": n_train, "validation": n_val, "test": n_test}


def make_split_manifest(
    records: List[dict],
    seed: int = DEFAULT_SEED,
    fractions: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Build a deterministic, group-isolated split manifest.

    Returns {"seed", "strategy", "splits": {split: [record_id, ...]},
             "groups": {split: [group_id, ...]}} with sorted ID lists.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    check_records(records)

    groups: Dict[str, List[dict]] = {}
    for rec in records:
        groups.setdefault(rec["group_id"], []).append(rec)

    time_based = "timestamp" in records[0]
    if time_based:
        strategy = "chronological-group"
        order = sorted(
            groups, key=lambda g: (min(r["timestamp"] for r in groups[g]), g)
        )
    else:
        strategy = "seeded-group-shuffle"
        order = sorted(groups)
        random.Random(seed).shuffle(order)

    counts = _group_counts(len(order), fractions)
    group_split: Dict[str, List[str]] = {}
    start = 0
    for split in SPLITS:
        group_split[split] = order[start:start + counts[split]]
        start += counts[split]

    manifest = {
        "seed": seed,
        "strategy": strategy,
        "splits": {
            split: sorted(r["record_id"] for g in group_split[split] for r in groups[g])
            for split in SPLITS
        },
        "groups": {split: sorted(group_split[split]) for split in SPLITS},
    }
    validate_manifest(manifest, records)
    return manifest


def validate_manifest(manifest: dict, records: List[dict]) -> None:
    """
    Reject a manifest that leaks: overlapping record IDs or groups between
    splits, unknown/missing records, or (with timestamps) a non-chronological
    order. Raises ValueError with the reason.
    """
    check_records(records)
    by_id = {r["record_id"]: r for r in records}
    splits = manifest["splits"]

    owner: Dict[str, str] = {}
    for split in SPLITS:
        for rid in splits[split]:
            if rid in owner:
                raise ValueError(
                    f"record_id {rid!r} appears in both {owner[rid]!r} and {split!r}"
                )
            if rid not in by_id:
                raise ValueError(f"unknown record_id {rid!r} in {split!r}")
            owner[rid] = split

    missing = sorted(set(by_id) - set(owner))
    if missing:
        raise ValueError(f"records not assigned to any split: {missing}")

    group_owner: Dict[str, str] = {}
    for rid, split in owner.items():
        gid = by_id[rid]["group_id"]
        if group_owner.setdefault(gid, split) != split:
            raise ValueError(
                f"group_id {gid!r} appears in both {group_owner[gid]!r} and {split!r}"
            )

    if "timestamp" in records[0]:
        prev_split, prev_max = None, None
        for split in SPLITS:
            times = [by_id[rid]["timestamp"] for rid in splits[split]]
            if prev_max is not None and min(times) < prev_max:
                raise ValueError(
                    f"split is not chronological: {split!r} starts before "
                    f"{prev_split!r} ends"
                )
            prev_split, prev_max = split, max(times)


def manifest_to_json(manifest: dict) -> str:
    """Canonical JSON text, so repeated runs give byte-identical manifests."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    """
    Build the manifest for the fixture, compare it with expected_manifest.json,
    and save inputs, expected vs actual, and tool versions.

    NOTE: Pipeline verification only — not evidence of real-world NEU
    steel surface defect performance.
    """
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture = load_fixture()
    first = make_split_manifest(fixture)
    second = make_split_manifest(fixture)
    expected = json.loads((LAB_DIR / "expected_manifest.json").read_text(encoding="utf-8"))

    (out / "fixture_input.json").write_text(
        json.dumps(fixture, indent=2), encoding="utf-8"
    )
    (out / "split_manifest.json").write_text(manifest_to_json(first), encoding="utf-8")

    result = {
        "note": (
            "Pipeline verification only. Synthetic 4×4 fixture — "
            "NOT the NEU steel surface defects dataset."
        ),
        "expected": expected,
        "actual": first,
        "repeated_run_identical": manifest_to_json(first) == manifest_to_json(second),
    }
    result["outcome"] = (
        "PASS" if first == expected and result["repeated_run_identical"] else "FAIL"
    )
    (out / "split_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
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

    print(f"[R42-T002] split manifest outcome: {result['outcome']}")
    for split in SPLITS:
        print(f"  {split:<10} records={first['splits'][split]} groups={first['groups'][split]}")
    return result


if __name__ == "__main__":
    run_and_save()
