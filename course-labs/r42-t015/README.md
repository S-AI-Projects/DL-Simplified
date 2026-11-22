# Devanagari Handwritten Character Preprocessing

## Overview

This lab demonstrates an explicit preprocessing pipeline for 4x4 grayscale images, preparing data for a future CNN workflow.

The included fixture is synthetic. It is not a real Devanagari handwriting dataset, and no CNN is trained in this task.

## Dataset Fixture

The fixture contains six synthetic images:
- 2 training records
- 2 validation records
- 2 test records
- Label 0: all-black image
- Label 1: white diagonal on a black background

Each record contains a unique record ID, group ID, split, label, and pixel matrix.

## Preprocessing Steps

1. Validate image dimensions (4x4).
2. Reject non-numeric, boolean, non-finite, and out-of-range pixels.
3. Convert pixel values to float32.
4. Normalize using pixel / 255.0.
5. Add a channel dimension, producing shape (4, 4, 1).
6. Validate record IDs, labels, splits, and group isolation.

## Split Isolation

A group ID cannot appear across different splits.

Preprocessing is stateless. It uses a fixed divisor of 255 and does not learn statistics from training, validation, or test data. There is no fitting step, so held-out data cannot influence preprocessing parameters.

## Requirements

- Python 3
- NumPy

No network access, GPU, or external dataset is required.

## Run the Pipeline

From the repository root:

    python3 course-labs/r42-t015/pipeline.py

Expected output:

    train: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]
    validation: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]
    test: images=(2, 4, 4, 1), dtype=float32, labels=[0, 1]

## Run Tests

From the repository root:

    python3 -m unittest discover -s course-labs/r42-t015 -p "test_*.py" -v

Tests cover dimensions, normalization, dtype, invalid pixels, duplicate IDs, missing splits, group leakage, and contaminated fixtures.

## Limitations

- The fixture is synthetic and very small.
- The labels represent artificial patterns, not actual Devanagari characters.
- No real-world handwriting dataset is included.
- No CNN is implemented or trained.
- No accuracy or model-performance claims are made.

This lab verifies preprocessing behavior only.
