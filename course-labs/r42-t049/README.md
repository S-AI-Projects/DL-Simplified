# Course Lab R42-T049: MobileNet vs ResNet - Input Perturbation Robustness

## Overview
This component tests input perturbation properties for the image interface. It verifies that record order permutations preserve invariant predictions when indexed by stable record IDs, while invalid inputs (such as shape mismatches or missing fields) are rejected with explicit errors.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 4x4 grayscale images (three all-zero images with label 0 and three diagonal 255 images with label 1).
**Note:** This is a CPU smoke fixture designed strictly for perturbation robustness and interface verification, NOT evidence of broad adversarial robustness or real-world performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t049 -p "test_*.py"
```
