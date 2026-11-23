# R42-T009 · FER-2013 facial expressions – CNN · Input-Perturbation Robustness

**Student:** Rohit Kumar Rai  
**Task:** R42-T009 · 9 of 10  
**Branch:** `R42-T009`  
**Closes:** [S-AI-Projects/DL-Simplified#9](https://github.com/S-AI-Projects/DL-Simplified/issues/9)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** FER-2013.
> It verifies **one narrow property** of the pipeline. It does **not** claim broad or adversarial robustness.

---

## Goal

Test one stated robustness property for the image interface, and show that one unsupported input is rejected.

## Stated property: record order (valid for images)

Shuffling or reversing the input records does not change the result once predictions are aligned by `record_id`. Each record
keeps its own ID. Images are not a time series, so reordering is a valid transformation here; no time order is randomised.

## Unsupported input: shape mismatch (rejected, no fallback)

A record is rejected with `ValueError` if its `pixels` field is missing or is not a flat list of 16 values (e.g. 15 values,
or a nested 4×4 list). The input is never padded, cropped or reshaped.

## Pipeline

- Split by group (`split.json`): train `g1` (r001, r004) / validation `g2` (r002, r005) / test `g3` (r003, r006).
- Fit a nearest-centroid classifier on train only (pixels / 255, squared Euclidean, tie → smallest label).
- Predict validation and test, returning `{record_id: label}`.

## Result

| Input order | Raw output order | Predictions aligned by ID |
|-------------|------------------|---------------------------|
| r001 … r006 (original) | r002, r003, r005, r006 | r002: 0, r003: 0, r005: 1, r006: 1 |
| r005, r001, r006, r004, r002, r003 (seed 7) | r005, r006, r002, r003 | r002: 0, r003: 0, r005: 1, r006: 1 |
| reversed | — | r002: 0, r003: 0, r005: 1, r006: 1 |

The property holds. Input with 15 pixels is rejected:
`record 'r006': unsupported input, shape mismatch: expected flat list of 16 pixels, got 15`.

---

## Files

```
course-labs/r42-t009/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── split.json            ← fixed group-level train/validation/test split
├── expected_result.json  ← expected robustness result
├── pipeline.py           ← input check, prediction by ID, seeded permutation
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T009/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json        ← input records, split, permutation seed
├── robustness_result.json    ← expected vs actual
├── tool_versions.json        ← Python / platform
├── unittest_output.txt       ← verbose test run (10 tests, exit_code=0)
└── mutation_check.txt        ← one expected prediction changed → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t009/pipeline.py              # writes work/R42-T009/ evidence
python3 -m unittest discover -s course-labs/r42-t009 -p "test_*.py"
```

## Done When

- [x] One valid transformation (record reordering) preserves the documented result after ID alignment
- [x] One unsupported input (shape mismatch) is rejected
- [x] No claim of broad adversarial robustness
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (10) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- Only record order is tested. Noise, rotations, brightness changes and adversarial perturbations are **not** tested.
- The permutation uses a fixed seed (7) so that the test is repeatable.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
