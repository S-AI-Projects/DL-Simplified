# Course Lab R42-T032: Dropout vs batch-norm ablation – CIFAR-10 Split Manifest

## Overview
This component implements a deterministic, leakage-free dataset split manifest for image records.
- Preserves group isolation strictly so no group ID appears across both train and test/validation sets.
- Generates reproducible, disjoint train/test splits.
- Rejects datasets with overlapping record IDs or inconsistent group assignments.

## Synthetic Fixture Notice
The fixture in `fixture_data.json` contains six synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real dataset performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t032 -p "test_*.py"
```
