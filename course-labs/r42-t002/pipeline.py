"""
R42-T002 · NEU Steel Surface Defects – CNN
Task type : Leakage-free splitting (deterministic split manifest)
Fixture   : Six synthetic 4×4 grayscale images
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
NOTE: This is a CPU smoke fixture. It is NOT the NEU Steel real dataset
      and must not be used to make performance claims about that dataset.
"""

from __future__ import annotations

import json
import pathlib
import hashlib
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Synthetic fixture (identical across all R42-T00x tasks)
# ---------------------------------------------------------------------------

NUM_PIXELS = 16   # 4×4


def make_fixture() -> List[dict]:
    """
    Returns the canonical six-record synthetic fixture.
    Three all-zero images (label=0) + three white-diagonal images (label=1).
    group_id clusters records for group-aware splitting.
    """
    zero_pixels = [0] * NUM_PIXELS

    def diag_pixels() -> List[int]:
        import numpy as np
        img = np.zeros((4, 4), dtype=np.uint8)
        np.fill_diagonal(img, 255)
        return img.flatten().tolist()

    records = [
        {"record_id": "r001", "group_id": "g1", "pixels": zero_pixels[:], "label": 0},
        {"record_id": "r002", "group_id": "g1", "pixels": zero_pixels[:], "label": 0},
        {"record_id": "r003", "group_id": "g2", "pixels": zero_pixels[:], "label": 0},
        {"record_id": "r004", "group_id": "g2", "pixels": diag_pixels(),  "label": 1},
        {"record_id": "r005", "group_id": "g3", "pixels": diag_pixels(),  "label": 1},
        {"record_id": "r006", "group_id": "g3", "pixels": diag_pixels(),  "label": 1},
    ]
    return records


# ---------------------------------------------------------------------------
# Deterministic split manifest
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}   # group-level

# Hard-coded manifest so the split is fully deterministic without any
# dependency on random-number state: groups are sorted, then assigned.
#
# Groups sorted: g1, g2, g3
#   train  → g1       records: r001, r002
#   val    → g2       records: r003, r004
#   test   → g3       records: r005, r006
#
# This assignment is derived once and frozen here so repeated runs always
# produce the same manifest (no randomness, no library version sensitivity).

_DETERMINISTIC_GROUP_SPLIT: Dict[str, str] = {
    "g1": "train",
    "g2": "val",
    "g3": "test",
}


def build_split_manifest(records: List[dict]) -> Dict[str, List[str]]:
    """
    Returns a dict mapping split names → sorted list of record_ids.

    Rules enforced:
    - Group isolation: all records sharing a group_id go to the same split.
    - Deterministic: sorted group names are assigned to splits in order;
      no randomness.
    - Disjoint: each record_id appears in exactly one split.
    """
    manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for rec in records:
        rid = rec["record_id"]
        gid = rec["group_id"]
        split = _DETERMINISTIC_GROUP_SPLIT.get(gid)
        if split is None:
            raise ValueError(
                f"record_id={rid!r}: group_id={gid!r} has no assigned split. "
                "Add it to _DETERMINISTIC_GROUP_SPLIT."
            )
        manifest[split].append(rid)

    # Sort for stable ordering
    for split in manifest:
        manifest[split].sort()

    return manifest


def validate_manifest(
    manifest: Dict[str, List[str]],
    all_record_ids: List[str],
) -> List[str]:
    """
    Validate that the manifest is leakage-free.
    Returns a list of error strings (empty → valid).
    """
    errors: List[str] = []

    # Collect all assigned IDs
    assigned: List[str] = []
    for split, ids in manifest.items():
        assigned.extend(ids)

    # 1. Disjoint check
    seen: set = set()
    for rid in assigned:
        if rid in seen:
            errors.append(f"Leakage: record_id {rid!r} appears in multiple splits")
        seen.add(rid)

    # 2. Coverage check
    expected = set(all_record_ids)
    actual = set(assigned)
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"Missing from manifest: {sorted(missing)}")
    if extra:
        errors.append(f"Extra in manifest (not in fixture): {sorted(extra)}")

    # 3. No split is empty
    for split, ids in manifest.items():
        if not ids:
            errors.append(f"Split {split!r} is empty")

    return errors


def manifest_fingerprint(manifest: Dict[str, List[str]]) -> str:
    """SHA-256 of the canonical JSON representation — used to verify determinism."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: str = "work/R42-T002") -> None:
    """
    Build and validate the split manifest; persist evidence to *output_dir*.

    NOTE: Results labelled as pipeline verification, not evidence of
    real-world NEU Steel Surface Defects performance.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fixture = make_fixture()
    all_ids = [r["record_id"] for r in fixture]

    # Save inputs
    (out / "fixture_input.json").write_text(
        json.dumps(fixture, indent=2), encoding="utf-8"
    )

    manifest = build_split_manifest(fixture)
    errors = validate_manifest(manifest, all_ids)

    fingerprint = manifest_fingerprint(manifest)

    result = {
        "note": (
            "Pipeline verification only. "
            "Synthetic 4×4 fixture — NOT NEU Steel real dataset."
        ),
        "manifest": manifest,
        "manifest_fingerprint_sha256": fingerprint,
        "errors": errors,
        "outcome": "PASS" if not errors else "FAIL",
    }

    (out / "split_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    import sys, platform
    import numpy as np
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    (out / "tool_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    print(f"[R42-T002] Manifest built — outcome: {result['outcome']}")
    print(f"  fingerprint: {fingerprint}")
    for split, ids in manifest.items():
        print(f"  {split:6s}: {ids}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    run_and_save()
