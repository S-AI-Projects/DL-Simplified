# Course Lab R42-T035: SMS Spam – LSTM Explicit Preprocessing

## Overview
This component implements explicit, leakage-free text tokenization and vocabulary encoding.
- Whitespace tokenization.
- Vocabulary built strictly from training records.
- Explicit unknown token `<unk>` with ID 0.
- Rejects transforming before fitting, and prevents fitting on held-out test data.

## Synthetic Fixture Notice
The fixture in `fixture_data.json` contains 6 synthetic text records (`t1` to `t6`).
**Note:** Strictly a CPU smoke fixture for pipeline verification, not real NLP corpus performance.

## Running Tests
```bash
python3 -m unittest discover -s course-labs/r42-t035 -p "test_*.py"
```
