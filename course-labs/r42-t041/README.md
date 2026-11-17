# Course Lab R42-T041: Hand Gesture Images - CNN Schema Validator

## Overview
This component implements a strict schema validator for input records intended for the Hand gesture images – CNN pipeline. It validates:
- `record_id`: Non-empty string, unique across records.
- `group_id`: Non-empty string grouping identifier.
- `pixels`: 4x4 matrix of numeric pixel intensity values in range [0, 255].
- `label`: Binary classification label (0 or 1).

Failures are reported explicitly per record without dropping any records silently.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for pipeline schema and contract verification, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests using standard unittest:
```bash
python3 -m unittest discover -s course-labs/r42-t041 -p "test_*.py"
```
