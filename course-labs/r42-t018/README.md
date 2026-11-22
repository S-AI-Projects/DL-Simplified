# R42-T018: Intel Image Scenes – Error Analysis

## Objective
Demonstrate a small, reproducible error-analysis example using synthetic grayscale images.

## Fixture
Six synthetic 4x4 grayscale images:
- Three all-zero images labelled 0.
- Three white-diagonal images labelled 1.
- Unique record IDs and group IDs.

This is a synthetic fixture, not the real Intel image-scenes dataset.

## Error analysis
The baseline predicts class 1 when any pixel is nonzero.

One incorrect prediction is deliberately injected:
- Record ID: diagonal-003
- Expected: 1
- Predicted: 0
- Reason: injected_prediction_error

The report contains one error and five correct predictions.

## Run

```bash
python3 course-labs/r42-t018/pipeline.py
