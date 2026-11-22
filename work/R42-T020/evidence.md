
# R42-T020 Evidence: Saved Text Pipeline Interface

## Environment

- Python: 3.14.6
- Branch: R42-T020
- Runtime: local macOS, CPU-only
- Dependencies: Python standard library only
- Network access and external datasets: not required

## Input

The pipeline uses six fictional records (`t1` through `t6`) from `course-labs/r42-t020/fixture.json`.

- Training records: t1, t2
- Validation records: t3, t4
- Test records: t5, t6

The fixture is synthetic and is used only for pipeline verification.

## Saved artifact

`course-labs/r42-t020/model.json`

- Format: JSON
- Schema version: 1.0
- Pipeline type: deterministic_bigram_demo
- Learned mappings from training records:
  - red -> apple
  - green -> apple
- No pickle serialization is used.

## Prediction verification

Input: `red`

| Check | Expected | Actual |
|---|---|---|
| Before save | apple | apple |
| After reload | apple | apple |
| Match | true | true |

Comparison uses exact string equality (tolerance 0).

## Automated tests

Command:

`python3 -m unittest discover -s course-labs/r42-t020 -p test_pipeline.py -v`

Result: 10 tests passed.

Coverage includes:
- Exact fixture records and labels
- Disjoint, complete data splits
- Training-only mapping
- Save/reload prediction consistency
- Unknown-token handling
- Invalid input and record schema
- Missing model mapping
- Unsupported schema version

## Limitations

This is a bounded pipeline-interface and JSON serialization smoke test. It is not a trained LSTM, does not use a real next-word corpus, and makes no real-world language-model performance claims.

