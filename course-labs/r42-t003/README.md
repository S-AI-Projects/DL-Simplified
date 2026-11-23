# R42-T003 · Grad-CAM explanations – any CNN · Minimal Baseline

**Student:** Rohit Kumar Rai  
**Task:** R42-T003 · 3 of 10  
**Branch:** `R42-T003`  
**Closes:** [S-AI-Projects/DL-Simplified#3](https://github.com/S-AI-Projects/DL-Simplified/issues/3)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** a real image dataset.
> All numbers below are **pipeline verification**, not evidence of real-world accuracy.

---

## Goal

Add the simplest honest baseline for the image classification contract: a **majority-class** classifier.
Its only parameter (the majority label) is fitted on the **train split only**; predictions and the chosen metric
(**accuracy**) are computed on the fixed held-out validation and test splits.

The baseline reads no pixels, so it has no spatial (Grad-CAM-style) attribution. It is the floor that a CNN, and any
explanation of that CNN, has to beat.

## Data contract

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | unique |
| `group_id` | `str` | assigns the record to a split (`split.json`) |
| `pixels` | `list[int]` | 16 values (4×4, row-major), 0–255 — not used by the baseline |
| `label` | `int` | `0` or `1` |

## Synthetic fixture and split

| record_id | group_id | pixels | label | split |
|-----------|----------|--------|-------|-------|
| r001 | g1 | all-zero | 0 | train |
| r002 | g1 | all-zero | 0 | train |
| r004 | g1 | white diagonal (255) | 1 | train |
| r005 | g2 | white diagonal (255) | 1 | validation |
| r003 | g3 | all-zero | 0 | test |
| r006 | g3 | white diagonal (255) | 1 | test |

The split is by group (`split.json`: train `g1`, validation `g2`, test `g3`), so no record or group is shared.
The groups are uneven on purpose, so training has a clear majority (two 0s, one 1).

## Result

| Item | Value |
|------|-------|
| Fitted on | r001, r002, r004 (train only) |
| Train label counts | 0 → 2, 1 → 1 |
| Majority label | **0** |
| Validation (r005) | truth [1], prediction [0], accuracy **0.0** |
| Test (r003, r006) | truth [0, 1], prediction [0, 0], accuracy **0.5** |

Saved in `expected_result.json`. Changing validation/test labels does not change the fitted parameter (tested).
Ties are broken towards the smallest label.

---

## Files

```
course-labs/r42-t003/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── split.json            ← fixed group-level train/validation/test split
├── expected_result.json  ← expected baseline result
├── pipeline.py           ← majority-class baseline + accuracy
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T003/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json    ← input records + split
├── baseline_result.json  ← expected vs actual
├── tool_versions.json    ← Python / platform
├── unittest_output.txt   ← verbose test run (14 tests, exit_code=0)
└── mutation_check.txt    ← expected test accuracy changed 0.5 → 0.75 → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t003/pipeline.py              # writes work/R42-T003/ evidence
python3 -m unittest discover -s course-labs/r42-t003 -p "test_*.py"
```

## Done When

- [x] Baseline parameters use only training data; predictions and accuracy are computed on the fixed held-out fixture
- [x] No real-dataset accuracy claim
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (14) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- With six records, the accuracy values are fixed by construction. They show the pipeline works; they say nothing about any real dataset.
- Accuracy on an empty split is rejected rather than reported as 0.
- No randomness, so no seed is needed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
