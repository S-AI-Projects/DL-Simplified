"""Deterministic split manifest generator and validator.

Project/section: Sign-language MNIST - CNN (text fixture).
Fixture: Six fictional records with IDs t1..t6.
Treat this as a pipeline fixture, not a real corpus.
"""

from typing import Any, Dict, List, Set


def create_deterministic_split(
    records: List[Dict[str, Any]],
    train_count: int = 4,
    val_count: int = 1,
    test_count: int = 1
) -> Dict[str, List[str]]:
    """Create a deterministic split manifest containing lists of record IDs.

    Sorted by record_id to ensure exact deterministic ordering across runs.
    """
    if len(records) != (train_count + val_count + test_count):
        raise ValueError(f"Record count {len(records)} does not match split sum {train_count + val_count + test_count}")

    sorted_records = sorted(records, key=lambda r: r["record_id"])
    record_ids = [r["record_id"] for r in sorted_records]

    train_ids = record_ids[:train_count]
    val_ids = record_ids[train_count:train_count + val_count]
    test_ids = record_ids[train_count + val_count:]

    return {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids
    }


def validate_split_manifest(manifest: Dict[str, List[str]]) -> None:
    """Validate that train, validation, and test record sets are strictly disjoint.

    Raises ValueError if overlap or duplicate ID is found.
    """
    train_set: Set[str] = set(manifest.get("train", []))
    val_set: Set[str] = set(manifest.get("validation", []))
    test_set: Set[str] = set(manifest.get("test", []))

    overlap_train_val = train_set & val_set
    if overlap_train_val:
        raise ValueError(f"Split leakage detected between train and validation: {overlap_train_val}")

    overlap_train_test = train_set & test_set
    if overlap_train_test:
        raise ValueError(f"Split leakage detected between train and test: {overlap_train_test}")

    overlap_val_test = val_set & test_set
    if overlap_val_test:
        raise ValueError(f"Split leakage detected between validation and test: {overlap_val_test}")

    for split_name in ("train", "validation", "test"):
        ids = manifest.get(split_name, [])
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate record IDs found within {split_name} split")
