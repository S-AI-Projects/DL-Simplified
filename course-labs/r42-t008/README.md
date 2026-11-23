# R42-T008 · EuroSAT land use – CNN · Error-Analysis Example

**Student:** Rohit Kumar Rai  
**Task:** R42-T008 · 8 of 10  
**Branch:** `R42-T008`  
**Closes:** [S-AI-Projects/DL-Simplified#8](https://github.com/S-AI-Projects/DL-Simplified/issues/8)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** EuroSAT.
> The report is **pipeline verification**, not evidence of real-world performance.

---

## Goal

Create a small error-analysis report for the image fixture. Each row has a stable record ID, expected value, predicted value
and reason category. One prediction is made wrong on purpose so the report can be tested.

## Pipeline

1. Split by group (`split.json`): train `g1` (r001, r004) / validation `g2` (r002, r005) / test `g3` (r003, r006).
2. Fit a nearest-centroid classifier on train only (pixels / 255, squared Euclidean, tie → smallest label), then predict
   validation and test. On its own the model gets all four held-out records right.
3. **Inject one wrong prediction:** `r006` (expected 1) is flipped to 0.
4. Build the report. It has **one row per wrong prediction only**, plus aggregate counts that are computed from the rows and
   cross-checked against them.

## Report columns

| Column | Meaning |
|--------|---------|
| `record_id` | stable fixture ID |
| `split` | `validation` or `test` |
| `expected` / `predicted` | true label / predicted label |
| `error_type` | `false_positive` (0 → 1) or `false_negative` (1 → 0) |
| `reason_category` | `injected_for_testing` (the deliberate error) or `nearer_to_wrong_centroid` (a genuine model error) |

## Result

| record_id | split | expected | predicted | error_type | reason_category |
|-----------|-------|----------|-----------|------------|-----------------|
| r006 | test | 1 | 0 | false_negative | injected_for_testing |

Aggregate: evaluated 4, correct 3, errors 1. By split: validation 0, test 1. By error type: false_negative 1. By reason:
injected_for_testing 1. Every aggregate sums to the number of detailed rows. The correct predictions (r002, r005, r003) do
not appear, and training records are not evaluated.

---

## Files

```
course-labs/r42-t008/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── split.json            ← fixed group-level train/validation/test split
├── expected_report.json  ← expected error report
├── pipeline.py           ← predictions, injection, report + aggregate check
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T008/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json        ← input records, split, injected ID
├── error_report_result.json  ← expected vs actual
├── tool_versions.json        ← Python / platform
├── unittest_output.txt       ← verbose test run (13 tests, exit_code=0)
└── mutation_check.txt        ← expected correct count changed 3 → 4 → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t008/pipeline.py              # writes work/R42-T008/ evidence
python3 -m unittest discover -s course-labs/r42-t008 -p "test_*.py"
```

## Done When

- [x] The injected wrong prediction appears in the report with the correct record ID (r006)
- [x] A correct prediction does not appear
- [x] Aggregate error counts match the detailed rows (checked in code and in tests)
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (13) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- The only error is the injected one; the model makes no genuine errors on this tiny fixture.
- Injection is refused for an unknown or training record, and for a prediction that is already wrong.
- No randomness, so no seed is needed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
