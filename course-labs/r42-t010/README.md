# R42-T010 · Food-101 subset – CNN · Saved-Pipeline Interface

**Student:** Rohit Kumar Rai  
**Task:** R42-T010 · 10 of 10  
**Branch:** `R42-T010`  
**Closes:** [S-AI-Projects/DL-Simplified#10](https://github.com/S-AI-Projects/DL-Simplified/issues/10)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** Food-101.
> All results are **pipeline verification**, not evidence of real-world performance.

---

## Goal

Save and reload the small image pipeline (configuration + parameters) as JSON, and expose a prediction function with schema
validation. An untrusted pickle is never loaded.

## Saved format (`saved_pipeline.json`)

```json
{
  "format": "r42-t010-nearest-centroid",
  "format_version": 1,
  "schema": {"required_fields": ["record_id", "pixels"], "pixels_length": 16,
             "pixel_min": 0, "pixel_max": 255, "labels": [0, 1]},
  "params": {"pixel_scale": 255, "centroids": {"0": [16 floats], "1": [16 floats]}}
}
```

- `fit_pipeline` learns one centroid per label from the **train** split only (`g1`: r001, r004).
- `save_pipeline` and `load_pipeline` accept `.json` paths only. Any other suffix (e.g. `.pkl`) is refused before the file is
  opened. There is no pickle anywhere in the code.
- `load_pipeline` fails clearly on:
  - wrong format;
  - version mismatch (`version mismatch: file has format_version 2, this code supports 1`);
  - schema mismatch;
  - malformed centroids.
- `predict(pipeline, records)` validates each input against the saved schema (a `record_id` string, plus 16 ints in
  [0, 255]). It returns `{record_id, label, scores}`, where scores are squared distances to each centroid and ties go to
  the smallest label.
- Inference needs **only the saved file and the input pixels**: no training data, labels or credentials.

## Tolerance

Labels before and after reload must be identical, and every score must agree within **1e-9** (absolute). The tests show
that a 1e-12 change passes and a 1e-3 change fails.

## Result

| record_id (split) | label | scores {0, 1} |
|-------------------|-------|---------------|
| r002 (validation) | 0 | 0.0, 4.0 |
| r005 (validation) | 1 | 4.0, 0.0 |
| r003 (test) | 0 | 0.0, 4.0 |
| r006 (test) | 1 | 4.0, 0.0 |

Before and after reload agree within tolerance.

---

## Files

```
course-labs/r42-t010/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── split.json            ← fixed group-level train/validation/test split
├── saved_pipeline.json   ← saved pipeline (JSON, fitted on train)
├── expected_result.json  ← expected before/after predictions
├── pipeline.py           ← fit / save / load / predict with schema checks
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T010/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json          ← input records + split
├── saved_pipeline.json         ← pipeline saved during the evidence run
├── saved_pipeline_result.json  ← expected vs actual
├── tool_versions.json          ← Python / platform
├── unittest_output.txt         ← verbose test run (15 tests, exit_code=0)
└── mutation_check.txt          ← one expected score changed → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t010/pipeline.py              # writes work/R42-T010/ evidence
python3 -m unittest discover -s course-labs/r42-t010 -p "test_*.py"
```

## Done When

- [x] Before and after predictions agree within an explicit tolerance (1e-9)
- [x] Schema and version mismatches fail clearly
- [x] No training data or credentials needed at inference time
- [x] JSON only; an untrusted pickle is never loaded
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (15) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- The saved model is a nearest-centroid classifier, not a CNN. The interface (save → load → validated predict) is what is tested.
- With one training image per label, each centroid equals that training image.
- No randomness, so no seed is needed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
