# R42-T019 Evidence

## Environment
- Python: 3.14.6
- Execution: Local CPU
- Dependencies: Python standard library
- Dataset: Six synthetic 4x4 grayscale images
- Random seed: 42

## Expected results
- Six records loaded successfully.
- Zero images predict label 0.
- Diagonal images predict label 1.
- Shuffling changes record order.
- Predictions remain identical after alignment by record ID.
- Missing pixels field is rejected.
- Invalid image shape and pixel values are rejected.

## Observed results
- Record count: 6
- Order changed: true
- Stable after alignment: true
- Unit tests: 10 passed
- Test status: OK

## Limitations
This is a synthetic input-order robustness test only. It does not use real UTKFace images, train a regression CNN, or measure age-estimation accuracy. It does not establish broad adversarial robustness.
