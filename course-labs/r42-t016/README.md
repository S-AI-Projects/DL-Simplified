# R42-T016: Bounded CPU Smoke Test

## Scope
This is an offline CPU smoke test. The task title refers to Energy Load Forecasting – LSTM, but its explicit acceptance criteria specify a six-record synthetic 4x4 image fixture. This implementation follows those criteria and does not implement an energy forecasting LSTM.

## Data contract
Each record contains:
- record_id: unique record identifier
- group_id: unique group identifier
- split: train, validation, or test
- pixels: 4x4 integer grayscale matrix, values 0 to 255
- label: binary label, 0 or 1

The fixture contains:
- Three all-zero images labelled 0
- Three white-diagonal images labelled 1
- Two records in each split
- Unique record IDs and group IDs

## Run
From the repository root:

    python3 course-labs/r42-t016/pipeline.py
    python3 -m unittest discover -s course-labs/r42-t016 -p "test_*.py" -v

## Smoke-test method
A deterministic nonzero-pixel baseline predicts class 1 if any pixel is nonzero, otherwise class 0.

This is not a trained neural network. It performs zero training updates. It uses seed 42 and is bounded to at most 32 samples and 20 update steps. No GPU, credentials, external dataset, or download is required.

## Environment
- Python 3.14.6
- NumPy 2.4.6

## Observed result
- Samples: 6
- Prediction shape: [6]
- Predictions: [0, 0, 1, 0, 1, 1]
- Mean squared error: 0.0
- Training updates: 0
- Unit tests: 10 passed

## Limitations
This synthetic fixture verifies pipeline execution only. It does not demonstrate real-world energy forecasting, LSTM performance, or generalisation. Zero loss reflects the deliberately separable synthetic examples.
