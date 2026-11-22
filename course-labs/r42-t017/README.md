# R42-T017: Reproducible Configuration

## Project
Ships in Satellite Images – CNN (configuration and pipeline smoke-test component).

## Objective
Demonstrate a reproducible, bounded, CPU-only image classification pipeline with a serializable configuration.

## Data
The fixture contains exactly six synthetic 4×4 grayscale images:
- Three all-zero images labelled 0.
- Three white-diagonal images labelled 1.
- Unique record IDs and group IDs.
- Explicit train, validation, and test splits.

These are artificial test patterns, not real satellite images or a real ships dataset.

## Configuration
`config.json` records:
- Schema version and random seed.
- Feature ordering.
- Explicit split record IDs.
- Algorithm name, threshold, sample limit, and step limit.

Unknown or incompatible configuration fields and settings are rejected.

## Pipeline
`pipeline.py` uses a deterministic nonzero-pixel baseline. It flattens pixels in row-major order and predicts class 1 when any pixel exceeds the configured threshold.

This is a smoke-test baseline, not a trained CNN.

## Run

```bash
python3 course-labs/r42-t017/pipeline.py
