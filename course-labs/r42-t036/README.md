# Course Lab R42-T036: Malaria Cell Images – CNN Bounded CPU Smoke Run

## Overview
This component implements a lightweight, bounded CPU smoke training run:
- Maximum 20 optimization steps / 1 epoch.
- Bounded input (at most 32 samples).
- Deterministic random seed.
- Verifies finite loss, expected output prediction shape, and step count bounds.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains 6 synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real clinical performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t036 -p "test_*.py"
```
