# Course Lab R42-T033: Variational autoencoder – Fashion-MNIST Minimal Baseline

## Overview
This component implements a majority-class and mean-reconstruction honest baseline for the Fashion-MNIST pipeline.
- Fits baseline parameters strictly on the training set.
- Evaluates on the held-out test set without data leakage.
- Requires no GPU or external service.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains 6 synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real dataset performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t033 -p "test_*.py"
```
