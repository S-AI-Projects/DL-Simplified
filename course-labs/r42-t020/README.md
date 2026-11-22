
# R42-T020: Saved Text Pipeline Interface

## Purpose

This task verifies a small, reproducible next-word prediction pipeline with JSON serialization, reload, and schema validation.

The fixture contains six fictional text records. It is synthetic test data, not a real language corpus.

## Pipeline design

This is a deterministic bigram demonstration, not a trained LSTM.

- Tokenization: lowercase whitespace splitting.
- Training: only records listed in the training split.
- Prediction: look up the final input token in the saved next-word mapping.
- Unknown tokens: return `<UNK>`.
- Serialization: JSON only; untrusted pickle files are never used.
- Runtime: Python standard library, CPU-only, offline.

## Data contract

Each fixture record contains exactly:

- `record_id`: non-empty string
- `text`: non-empty string
- `label`: integer, either 0 or 1

The fixture has six records with IDs `t1` through `t6`.

Splits:
- Train: `t1`, `t2`
- Validation: `t3`, `t4`
- Test: `t5`, `t6`

The splits are disjoint and cover all six records.

## Inference contract

`load_pipeline(path)` loads and validates a saved JSON pipeline.

`predict_next(model, text)` accepts a non-empty string and returns a next-word string. It does not require access to the fixture or training records.

The saved model includes a schema version, pipeline type, parameters, and next-word mapping.

Unsupported schema versions and malformed model structures fail with clear `ValueError` messages.

## Run

From the repository root:

```bash
python3 course-labs/r42-t020/pipeline.py
python3 -m unittest discover -s course-labs/r42-t020 -p test_pipeline.py -v
```

Running the pipeline creates `model.json` in this directory.

## Verification result

The demonstration predicts `apple` for input `red` both before saving and after reloading.

The comparison is exact string equality (tolerance 0).

The test suite verifies the fixture, split separation, training-only mapping, save/reload consistency, unknown tokens, invalid input, and schema failures.

## Limitations

This is an interface and serialization smoke test only. It does not train or evaluate an LSTM, use a real next-word corpus, or establish real-world language-model accuracy.

