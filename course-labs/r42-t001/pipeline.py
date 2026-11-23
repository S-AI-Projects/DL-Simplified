"""
R42-T001 · Fashion-MNIST – CNN
Task type : Dataset contract (schema validator)
Fixture   : Six synthetic 4×4 grayscale images
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
NOTE: This is a CPU smoke fixture. It is NOT the Fashion-MNIST real dataset
      and must not be used to make performance claims about that dataset.
"""

from __future__ import annotations

import json
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
IMAGE_ROWS = 4
IMAGE_COLS = 4
NUM_PIXELS = IMAGE_ROWS * IMAGE_COLS          # 16
VALID_LABELS = {0, 1}
PIXEL_DTYPE = int                             # integer 0-255


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def make_fixture() -> List[dict]:
    """
    Returns the canonical six-record synthetic fixture.
    Three all-zero images (label=0) + three white-diagonal images (label=1).
    group_id clusters records for group-aware splitting.
    """
    zero_pixels = [0] * NUM_PIXELS

    def diag_pixels() -> List[int]:
        img = np.zeros((IMAGE_ROWS, IMAGE_COLS), dtype=np.uint8)
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
# Schema validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationFailure:
    record_id: Optional[str]
    field: str
    reason: str


def validate_record(record: Any, seen_ids: set) -> List[ValidationFailure]:
    """
    Validate one record against the schema.
    Returns a (possibly empty) list of ValidationFailure objects.
    Records are NEVER silently dropped — failures are always reported.
    """
    failures: List[ValidationFailure] = []

    # --- record_id ---
    rid = record.get("record_id") if isinstance(record, dict) else None
    if rid is None:
        failures.append(ValidationFailure(
            record_id=None, field="record_id",
            reason="missing or null record_id"
        ))
    elif not isinstance(rid, str) or rid.strip() == "":
        failures.append(ValidationFailure(
            record_id=rid, field="record_id",
            reason=f"record_id must be a non-empty string, got {type(rid).__name__!r}"
        ))
    elif rid in seen_ids:
        failures.append(ValidationFailure(
            record_id=rid, field="record_id",
            reason=f"duplicate record_id {rid!r}"
        ))
    else:
        seen_ids.add(rid)

    # --- group_id ---
    gid = record.get("group_id") if isinstance(record, dict) else None
    if gid is None:
        failures.append(ValidationFailure(
            record_id=rid, field="group_id",
            reason="missing or null group_id"
        ))
    elif not isinstance(gid, str) or gid.strip() == "":
        failures.append(ValidationFailure(
            record_id=rid, field="group_id",
            reason=f"group_id must be a non-empty string, got {type(gid).__name__!r}"
        ))

    # --- pixels ---
    pixels = record.get("pixels") if isinstance(record, dict) else None
    if pixels is None:
        failures.append(ValidationFailure(
            record_id=rid, field="pixels",
            reason="missing or null pixels"
        ))
    elif not isinstance(pixels, list):
        failures.append(ValidationFailure(
            record_id=rid, field="pixels",
            reason=f"pixels must be a list, got {type(pixels).__name__!r}"
        ))
    elif len(pixels) != NUM_PIXELS:
        failures.append(ValidationFailure(
            record_id=rid, field="pixels",
            reason=(
                f"pixels length must be {NUM_PIXELS} (4×4), got {len(pixels)}"
            )
        ))
    else:
        for i, px in enumerate(pixels):
            if not isinstance(px, int):
                failures.append(ValidationFailure(
                    record_id=rid, field=f"pixels[{i}]",
                    reason=f"pixel value must be int, got {type(px).__name__!r}"
                ))
                break
            if not (0 <= px <= 255):
                failures.append(ValidationFailure(
                    record_id=rid, field=f"pixels[{i}]",
                    reason=f"pixel value {px} out of range [0, 255]"
                ))
                break

    # --- label ---
    label = record.get("label") if isinstance(record, dict) else None
    if label is None:
        failures.append(ValidationFailure(
            record_id=rid, field="label",
            reason="missing or null label"
        ))
    elif not isinstance(label, int):
        failures.append(ValidationFailure(
            record_id=rid, field="label",
            reason=f"label must be int, got {type(label).__name__!r}"
        ))
    elif label not in VALID_LABELS:
        failures.append(ValidationFailure(
            record_id=rid, field="label",
            reason=f"label {label!r} not in valid set {VALID_LABELS}"
        ))

    return failures


def validate_dataset(records: List[Any]) -> List[ValidationFailure]:
    """Validate every record; collect all failures without dropping records."""
    seen_ids: set = set()
    all_failures: List[ValidationFailure] = []
    for record in records:
        all_failures.extend(validate_record(record, seen_ids))
    return all_failures


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: str = "work/R42-T001") -> None:
    """
    Run validation on the canonical fixture, print results, and persist
    evidence (inputs + expected vs actual) to *output_dir*.

    NOTE: Results labelled as pipeline verification, not evidence of
    real-world Fashion-MNIST performance.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fixture = make_fixture()

    # Save inputs
    (out / "fixture_input.json").write_text(
        json.dumps(fixture, indent=2), encoding="utf-8"
    )

    # Run validation — expect zero failures on clean fixture
    failures = validate_dataset(fixture)

    result = {
        "note": (
            "Pipeline verification only. "
            "Synthetic 4×4 fixture — NOT Fashion-MNIST real dataset."
        ),
        "records_validated": len(fixture),
        "failures": [
            {"record_id": f.record_id, "field": f.field, "reason": f.reason}
            for f in failures
        ],
        "outcome": "PASS" if not failures else "FAIL",
    }

    (out / "validation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    import sys, platform
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    (out / "tool_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    print(f"[R42-T001] Validated {len(fixture)} records — outcome: {result['outcome']}")
    if failures:
        for f in failures:
            print(f"  FAIL  record_id={f.record_id!r}  field={f.field!r}  reason={f.reason}")


if __name__ == "__main__":
    run_and_save()
