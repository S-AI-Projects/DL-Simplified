# Course Lab R42-T047: Yoga Pose Images - Reproducible Configuration

## Overview
This component implements a serializable and validated configuration specification for the image fixture. It captures schema version, seed, feature ordering, split IDs, and algorithm settings while strictly rejecting unknown keys or incompatible input shapes.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for configuration integrity and pipeline verification, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t047 -p "test_*.py"
```
