# Course Lab R42-T050: MNIST MLP - Saved Pipeline Interface

## Overview
This component implements a safe, JSON-based model serialization and reloading interface for a small MLP classifier. It exposes an inference prediction function with input schema verification, rejecting version mismatches, corrupt files, or malformed inference inputs without requiring untrusted pickle deserialization.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for serialization and interface testing, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t050 -p "test_*.py"
```
