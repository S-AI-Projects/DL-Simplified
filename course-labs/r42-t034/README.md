# Course Lab R42-T034: Speech Commands – Audio CNN Evaluation Metric

## Overview
This component implements and verifies classification and regression metrics for audio processing fixtures:
- Explicit zero-denominator handling for precision, recall, and F1 score.
- Exact match on canonical test cases:
  - truth `[1, 1, 0, 0]` with prediction `[1, 0, 1, 0]` produces precision = recall = 0.5.
  - truth `[1, 2, 3]` with prediction `[1, 3, 2]` produces MAE = 2/3 (0.6667).

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six synthetic 8-sample audio waveforms (three zero arrays, three alternating -1/1 arrays).
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real acoustic performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t034 -p "test_*.py"
```
