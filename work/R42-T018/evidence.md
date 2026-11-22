# R42-T018 Evidence

## Environment
- Python: 3.14.6
- Dependencies: Python standard library only
- Execution: Offline, CPU-only
- Seed: 42

## Input
- course-labs/r42-t018/fixture.json

## Pipeline results
- Total records: 6
- Correct predictions: 5
- Errors: 1

Injected error:
- Record ID: diagonal-003
- Expected: 1
- Predicted: 0
- Reason: injected_prediction_error

## Tests
Command:
python3 -m unittest discover -s course-labs/r42-t018 -p test_pipeline.py

Result: 11 tests passed.

## Limitations
Synthetic pipeline verification only. No real Intel dataset or trained CNN is used.
