# Course Lab R42-T045: Garbage Classification - Explicit Preprocessing

## Overview
This component implements an explicit preprocessing pipeline for the synthetic image dataset. It scales raw pixel values from [0, 255] to [0.0, 1.0] and ensures an explicit channel axis of shape (4, 4, 1), strictly rejecting invalid shapes and out-of-range pixels.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for pipeline verification and contract testing, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t045 -p "test_*.py"
```
