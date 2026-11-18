# Course Lab R42-T043: Oxford-IIIT Pets - Minimal Baseline

## Overview
This component implements a majority-class baseline classifier and metric evaluation (accuracy and macro F1) for the image contract. Baseline parameters are fit exclusively using training split data.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for baseline logic and pipeline verification, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t043 -p "test_*.py"
```
