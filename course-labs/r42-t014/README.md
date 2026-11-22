
# R42-T014: Evaluation Metric Verification

## Objective
Verify classification metrics and regression MAE using a small synthetic waveform fixture.

## Dataset
- Six synthetic waveform records.
- Eight samples per record.
- Sample rate: 8000 Hz.
- Three records labelled 0 and three labelled 1.
- This is not the real ESC-50 dataset.

## Metrics
- Binary precision
- Binary recall
- Mean Absolute Error (MAE)

## Expected Results
- True positives: 1
- False positives: 1
- False negatives: 1
- Precision: 0.5
- Recall: 0.5
- Regression MAE: 0.6666667
- Zero denominator returns 0.0.

## Run
```bash
python3 course-labs/r42-t014/pipeline.py
python3 -m unittest discover -s course-labs/r42-t014 -p "test_*.py" -v
```

## Limitations
This verifies metric calculations on synthetic data only. It does not train a CNN, process spectrograms, or measure real ESC-50 performance.
