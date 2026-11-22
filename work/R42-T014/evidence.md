
# R42-T014 Evidence

## Environment
- Python: Record using `python3 --version`
- OS: macOS
- Dependencies: Python standard library only
- Execution: Offline, CPU-only

## Input
- Six synthetic waveform records.
- Eight samples per record.
- Sample rate: 8000 Hz.
- Not the real ESC-50 dataset.

## Actual Results
- True positives: 1
- False positives: 1
- False negatives: 1
- Precision: 0.5
- Recall: 0.5
- Regression MAE: 0.6666667
- Unit tests: 11 passed, 0 failed.

## Commands Run
- `python3 course-labs/r42-t014/pipeline.py`
- `python3 -m unittest discover -s course-labs/r42-t014 -p "test_*.py" -v`

## Limitations
These results verify metric calculations on a synthetic fixture only. They are not real ESC-50 or CNN performance claims.
