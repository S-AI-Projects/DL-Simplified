"""Deterministic leakage-free splitter for R42-T032."""
import hashlib
from typing import Any, Dict, List, Set

def split_manifest(records: List[Dict[str, Any]], test_ratio: float = 0.5) -> Dict[str, List[str]]:
    # Validate uniqueness
    seen_ids = set()
    group_to_ids: Dict[str, List[str]] = {}
    for r in records:
        rid = r["record_id"]
        gid = r["group_id"]
        if rid in seen_ids:
            raise ValueError(f"Duplicate record ID found: {rid}")
        seen_ids.add(rid)
        group_to_ids.setdefault(gid, []).append(rid)

    # Deterministic group assignment using SHA-256 hash
    train_ids = []
    test_ids = []
    
    sorted_groups = sorted(group_to_ids.keys())
    for gid in sorted_groups:
        h = int(hashlib.sha256(gid.encode("utf-8")).hexdigest(), 16) % 100
        if h < (test_ratio * 100):
            test_ids.extend(group_to_ids[gid])
        else:
            train_ids.extend(group_to_ids[gid])
            
    # Guarantee both splits non-empty for test suite if multiple groups exist
    if (not train_ids or not test_ids) and len(sorted_groups) > 1:
        test_ids = list(group_to_ids[sorted_groups[0]])
        train_ids = []
        for g in sorted_groups[1:]:
            train_ids.extend(group_to_ids[g])

    train_set = set(train_ids)
    test_set = set(test_ids)
    if train_set.intersection(test_set):
        raise ValueError("Data leakage detected: overlapping record IDs across splits!")

    return {
        "train": sorted(train_ids),
        "test": sorted(test_ids)
    }
