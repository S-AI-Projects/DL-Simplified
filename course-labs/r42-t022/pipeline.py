"""
R42-T022: Deterministic split manifest for image fixture.
Preserves group isolation (group_id) across train, val, and test splits.
Writes exact record IDs to each split.
"""
from typing import Any, Dict, List, Set

def create_deterministic_split(records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    # Group records by group_id
    groups: Dict[str, List[str]] = {}
    for r in records:
        gid = r["group_id"]
        groups.setdefault(gid, []).append(r["record_id"])
    
    sorted_groups = sorted(groups.keys())
    # Deterministic assignment of groups to splits: grp_a -> train, grp_b -> val, grp_c -> test
    train_ids = []
    val_ids = []
    test_ids = []
    
    for i, gid in enumerate(sorted_groups):
        if i % 3 == 0:
            train_ids.extend(groups[gid])
        elif i % 3 == 1:
            val_ids.extend(groups[gid])
        else:
            test_ids.extend(groups[gid])
            
    train_ids.sort()
    val_ids.sort()
    test_ids.sort()
    
    # Check disjointness
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
        raise ValueError("Split leakage detected: record IDs overlap across splits!")
        
    return {
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "total_records": len(records),
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    }

def verify_manifest(manifest: Dict[str, Any], records: List[Dict[str, Any]]) -> bool:
    rec_to_grp = {r["record_id"]: r["group_id"] for r in records}
    train_grps = {rec_to_grp[rid] for rid in manifest["train"]}
    val_grps = {rec_to_grp[rid] for rid in manifest["val"]}
    test_grps = {rec_to_grp[rid] for rid in manifest["test"]}
    if (train_grps & val_grps) or (train_grps & test_grps) or (val_grps & test_grps):
        return False
    return True
