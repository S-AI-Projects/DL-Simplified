# Task 15 Evidence — Devanagari Image Preprocessing

## Environment
- Python: 3.14.6
- NumPy: 2.4.6
- Platform: macOS
- Execution: CPU-only

## Implementation
- Created a synthetic 4x4 grayscale fixture containing six records.
- Implemented strict image dimension and pixel-value validation.
- Converted images to float32.
- Normalized pixel values from 0–255 to 0–1.
- Added an explicit channel dimension.
- Validated record IDs, labels, splits, and group isolation.
- Used stateless preprocessing with a fixed divisor of 255.

## Pipeline Output

Command:

    python3 course-labs/r42-t015/pipeline.py

Output:

    train: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]
    validation: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]
    test: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]

## Automated Tests

Command:

    python3 -m unittest discover -s course-labs/r42-t015 -p "test_*.py" -v

Result:
- 16 tests passed.
- 0 failures.
- Covered normalization, shape, dtype, invalid pixels, duplicate IDs, missing splits, group leakage, and contaminated fixture rejection.

## Limitations

The fixture contains synthetic patterns only. It is not a real Devanagari handwriting dataset. No CNN training, real-world accuracy, or model-performance claims are included.

This task verifies preprocessing behavior only.
