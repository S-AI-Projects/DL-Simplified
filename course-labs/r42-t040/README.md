# Course Lab R42-T040: Image Colourisation – Autoencoder Saved Pipeline

## Overview
This component implements a safe parameter save and reload interface:
- Saves and loads pipeline state via safe JSON (strictly avoiding untrusted pickles).
- Verifies that pre-save and post-reload predictions agree within explicit numeric tolerance (1e-6).
- Enforces schema version and input dimensionality validation at inference time.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains 6 synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real clinical performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t040 -p "test_*.py"
```
