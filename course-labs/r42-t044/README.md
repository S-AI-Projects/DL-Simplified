# Course Lab R42-T044: SVHN House Numbers - Evaluation Metric Verification

## Overview
This component implements an explicit binary evaluation metric calculator (Precision, Recall, F1, Accuracy) with clearly defined zero-denominator behavior. It verifies the canonical case:
`truth=[1, 1, 0, 0]` and `prediction=[1, 0, 1, 0]` yielding `precision=0.5` and `recall=0.5`.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for metric and pipeline verification, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t044 -p "test_*.py"
```
