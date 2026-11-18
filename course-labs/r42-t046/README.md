# Course Lab R42-T046: Weather Images - Bounded CPU Smoke Run

## Overview
This component implements a lightweight, deterministic CPU smoke training loop using a tiny linear classifier. It runs at most 20 update steps (1 bounded pass over the 6 fixture samples) with a fixed seed, finite loss assertions, and zero external or GPU dependencies.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for pipeline interface and bounded runtime verification, NOT evidence of real-world or clinical performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t046 -p "test_*.py"
```
