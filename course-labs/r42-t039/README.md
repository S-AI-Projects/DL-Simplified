# Course Lab R42-T039: Chest X-ray Pneumonia – CNN Perturbation Test

## Overview
This component tests interface robustness properties for image fixtures:
- Record-order permutation preserves outputs once aligned by record ID.
- Rejects malformed input structures and shape mismatches.
- Does not claim broad adversarial clinical robustness.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains 6 synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real clinical performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t039 -p "test_*.py"
```
