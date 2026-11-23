# R42-T002 · NEU steel surface defects – CNN · Leakage-free Splitting

**Student:** Rohit Kumar Rai  
**Task:** R42-T002 · 2 of 10  
**Branch:** `R42-T002`  
**Closes:** [S-AI-Projects/DL-Simplified#2](https://github.com/S-AI-Projects/DL-Simplified/issues/2)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the NEU steel surface defects dataset and must not be used to make performance claims about NEU or any real-world task.
> All results below are **pipeline verification**, not evidence of real-world performance.

---

## Goal

Build a **deterministic split manifest** for the image fixture that writes the exact record IDs to train / validation / test
and keeps every `group_id` inside a single split, so no group leaks between splits.

## Data contract

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | non-empty, unique across all records |
| `group_id` | `str` | non-empty; all records of a group go to the same split |
| `pixels` | `list[int]` | 16 values (4×4, row-major), 0–255 — carried as-is, not used for splitting |
| `label` | `int` | `0` or `1` — carried as-is, not used for splitting |
| `timestamp` *(optional)* | number | if present on **all** records, the split is chronological |

## Synthetic fixture (`fixture.json`)

| record_id | group_id | pixels | label |
|-----------|----------|--------|-------|
| r001 | g1 | all-zero (4×4) | 0 |
| r002 | g2 | all-zero (4×4) | 0 |
| r003 | g3 | all-zero (4×4) | 0 |
| r004 | g1 | white diagonal, 255 (4×4) | 1 |
| r005 | g2 | white diagonal, 255 (4×4) | 1 |
| r006 | g3 | white diagonal, 255 (4×4) | 1 |

Each group holds one image of each label, so every split contains both classes.

## Split method (`pipeline.py`)

1. Check inputs: every record needs a unique `record_id` and a `group_id`; duplicates or missing values raise `ValueError`.
2. Order the groups:
   - no timestamps → sort group IDs, then shuffle with `random.Random(seed)` (default seed `42`);
   - timestamps present → order groups by their earliest timestamp (oldest first).
3. Assign whole groups using fractions train 0.6 / validation 0.2 / test 0.2 (at least one group per split; needs ≥ 3 groups).
4. `validate_manifest` rejects any manifest where a record ID or a group appears in more than one split, a record is
   missing or unknown, or (with timestamps) a later split starts before an earlier one ends.
5. `manifest_to_json` writes sorted, canonical JSON, so repeated runs are byte-identical.

## Result (seed 42)

| Split | record IDs | group |
|-------|-----------|-------|
| train | r002, r005 | g2 |
| validation | r001, r004 | g1 |
| test | r003, r006 | g3 |

Saved as `expected_manifest.json`. The three record sets are disjoint, each group is in exactly one split,
and repeated runs (and reversed input order) give the same manifest.

---

## Files

```
course-labs/r42-t002/
├── fixture.json            ← literal synthetic fixture (simulated data)
├── expected_manifest.json  ← expected split manifest for seed 42
├── pipeline.py             ← split manifest + leakage checks
├── test_pipeline.py        ← unit tests
└── README.md               ← this file
work/R42-T002/               ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json      ← input
├── split_manifest.json     ← actual manifest
├── split_result.json       ← expected vs actual + repeated-run check
├── tool_versions.json      ← Python / platform
├── unittest_output.txt     ← verbose test run (19 tests, exit_code=0)
└── mutation_check.txt      ← one expected record ID deliberately changed → FAILED, exit_code=1
```

## How to Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. No dependencies to install (standard library only)

# 3. Build the manifest (writes work/R42-T002/ evidence at the repo root)
python3 course-labs/r42-t002/pipeline.py

# 4. Run the automated check (exits non-zero if any assertion fails)
python3 -m unittest discover -s course-labs/r42-t002 -p "test_*.py"
```

## Done When

- [x] Train/validation/test record sets are disjoint
- [x] Repeated runs produce the same manifest
- [x] Intentionally overlapping record IDs / groups are rejected
- [x] Time-series splits are chronological (tested with a timestamped copy of the fixture; the fixture itself has no time field)
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (19) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- Splits whole groups only; with 3 groups, each split gets exactly one group (2 records). Fractions are applied to
  group counts, not record counts.
- Does not train a CNN or compute any metric.
- The shuffle relies on Python's `random.Random`, which is stable across Python 3 versions for a fixed seed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
