
# R42-T012 Validation Summary

## Task
Prevent data leakage between train, validation, and test splits.

## Input
- Six synthetic 4x4 grayscale image records.
- Three distinct groups.
- Two records per group.
- Synthetic data only; not a real cassava dataset.

## Results
- All six records assigned exactly once: PASS
- No group overlap between splits: PASS
- Fixed seed produces reproducible output: PASS
- Duplicate record IDs rejected: PASS
- Invalid pixel values rejected: PASS
- Fewer than three groups rejected: PASS
- Expected and actual manifests match: PASS

## Environment
- Python 3.14.6
- Standard library only
- CPU-only execution

## Overall Status
PASS — All six unit tests passed.

