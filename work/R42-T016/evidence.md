# R42-T016 Evidence

## Environment
- Python: 3.14.6
- NumPy: 2.4.6
- Device: CPU
- Data: local synthetic fixture; no external download

## Input
- Fixture: course-labs/r42-t016/fixture.json
- Image shape: 4x4
- Pixel range: 0-255
- Records: 6
- Labels: three 0 and three 1
- Splits: 2 train, 2 validation, 2 test
- Record IDs and group IDs: unique

## Expected and actual result
- Expected sample count: 6; actual: 6
- Expected prediction shape: [6]; actual: [6]
- Expected predictions: [0, 0, 1, 0, 1, 1]
- Actual predictions: [0, 0, 1, 0, 1, 1]
- Mean squared error: 0.0
- Training updates: 0
- Unit tests: 10 passed

## Test command
python3 -m unittest discover -s course-labs/r42-t016 -p "test_*.py" -v

Observed result: 10 tests ran; all passed.

## Limitations
This is pipeline verification on six synthetic 4x4 images only. The implementation is a deterministic nonzero-pixel baseline, not a trained LSTM. It does not use a real energy dataset or establish real-world forecasting performance. Zero loss reflects the deliberately separable synthetic fixture and must not be interpreted as real-world model accuracy.
