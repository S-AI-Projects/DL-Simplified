# R42-T019: Input Perturbation Robustness Check

## Objective

Test whether changing the order of image records affects predictions when results are aligned by stable record ID.

## Fixture

- Six synthetic 4x4 grayscale images.
- Three all-zero images with label 0.
- Three diagonal-pattern images with label 1.
- Unique record IDs and group IDs.
- Pixel values range from 0 to 255.

This is a bounded synthetic smoke test. It is not the real UTKFace age-estimation dataset and does not train or evaluate an age-estimation regression CNN.

## Method

1. Load and validate the fixture.
2. Generate deterministic baseline predictions using a nonzero-pixel rule.
3. Shuffle the records using seed 42.
4. Generate predictions in shuffled order.
5. Compare the prediction mappings by record ID.

## Expected result

- Record order changes after shuffling.
- Predictions remain identical after alignment by record ID.
- A record missing its `pixels` field is rejected.
- Invalid image shapes and pixel values are rejected.

## Run

```bash
python3 course-labs/r42-t019/pipeline.py
python3 -m unittest discover -s course-labs/r42-t019 -p test_pipeline.py
