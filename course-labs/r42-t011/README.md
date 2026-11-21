# R42-T011: Text Pipeline Validation

## Overview
This project contains basic validation functions for text records.

## Record Format
Each record should contain:
- `record_id`: A non-empty string
- `text`: A non-empty text string
- `label`: A numeric value

## Validation
The pipeline checks:
- Required fields are present
- Record ID and text are not empty
- Label is numeric
- Record IDs are not duplicated

## Run Tests
From the repository root, run:

```bash
python3 -m unittest discover -s course-labs/r42-t011 -p "test_*.py"