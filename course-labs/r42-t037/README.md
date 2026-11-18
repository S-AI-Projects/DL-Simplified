# Course Lab R42-T037: Denoising Autoencoder – MNIST Reproducible Config

## Overview
This component implements a serializable configuration schema for the MNIST denoising autoencoder pipeline:
- Validates schema version, seed, feature ordering, and model architecture hyperparameters.
- Rejects unknown attributes or contradictory parameters (e.g. noise_level < 0 or negative epochs).
- Reloaded configuration strictly reproduces identical pipeline operations.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains 6 synthetic 4x4 grayscale images.
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real dataset performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t037 -p "test_*.py"
```
