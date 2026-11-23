# R42-T004 · Horses vs humans – CNN · Metric Correctness

**Student:** Rohit Kumar Rai  
**Task:** R42-T004 · 4 of 10  
**Branch:** `R42-T004`  
**Closes:** [S-AI-Projects/DL-Simplified#4](https://github.com/S-AI-Projects/DL-Simplified/issues/4)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images) and **simulated, fixed predictions**.
> It is **not** the Horses vs humans dataset. All numbers are **pipeline verification**, not real-world performance.

---

## Goal

Implement the evaluation for the binary image fixture, with explicitly defined averaging and zero-denominator behaviour,
and cross-check it independently.

## Metric definitions (labels 0/1, positive class = 1)

For a class *c*: TP = truth = c and prediction = c; FP = truth ≠ c and prediction = c; FN = truth = c and prediction ≠ c.

| Metric | Formula |
|--------|---------|
| precision | TP / (TP + FP) |
| recall | TP / (TP + FN) |
| F1 | 2·TP / (2·TP + FP + FN) |
| accuracy | correct / N |

**Averaging**

- `average="binary"` (default): the metric for the positive class 1 only.
- `average="macro"`: the unweighted mean of the per-class metric over classes 0 and 1.

**Zero denominator**

- If a denominator is 0, that class's metric is `zero_division`: `0.0` (default) or `1.0`.
- `zero_division="raise"` raises `ZeroDivisionError` instead.
- With `macro`, the replacement is applied per class before averaging.
- Accuracy (or any metric) on zero records, mismatched lengths, or labels other than 0/1 (including `bool`) raise `ValueError`.

## Required example (from the task)

truth `[1, 1, 0, 0]`, prediction `[1, 0, 1, 0]` → TP = 1, FP = 1, FN = 1 → **precision = recall = 0.5**
(also F1 = 0.5, macro precision = 0.5, accuracy = 0.5).

## Fixture evaluation

| record_id | group_id | pixels | label (truth) | simulated prediction |
|-----------|----------|--------|---------------|----------------------|
| r001 | g1 | all-zero | 0 | 0 |
| r002 | g2 | all-zero | 0 | 1 |
| r003 | g3 | all-zero | 0 | 0 |
| r004 | g1 | white diagonal (255) | 1 | 1 |
| r005 | g2 | white diagonal (255) | 1 | 0 |
| r006 | g3 | white diagonal (255) | 1 | 1 |

The predictions are fixed and simulated (`predictions.json`), not the output of a trained model. No model is trained here,
so all six records form the **test set only**; there is no train/validation data in this component.

Result: TP = 2, FP = 1, FN = 1 → accuracy, precision, recall and F1 (binary and macro) are all **2/3**.

## Independent cross-check

`test_pipeline.py` contains a separate set-based reference implementation. It is compared with `pipeline.py` on **all 256**
length-4 truth/prediction pairs, for both averaging modes and both `zero_division` values.

---

## Files

```
course-labs/r42-t004/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── predictions.json      ← fixed simulated predictions per record
├── expected_result.json  ← expected metric values
├── pipeline.py           ← metrics + fixture evaluation
├── test_pipeline.py      ← unit tests + independent reference
└── README.md             ← this file
work/R42-T004/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json    ← input records + predictions
├── metric_result.json    ← expected vs actual
├── tool_versions.json    ← Python / platform
├── unittest_output.txt   ← verbose test run (18 tests, exit_code=0)
└── mutation_check.txt    ← expected precision changed 0.5 → 0.75 → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t004/pipeline.py              # writes work/R42-T004/ evidence
python3 -m unittest discover -s course-labs/r42-t004 -p "test_*.py"
```

## Done When

- [x] Classification: truth [1,1,0,0] / prediction [1,0,1,0] → precision = recall = 0.5
- [x] Averaging and zero-denominator behaviour are defined and tested
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (18) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- Binary labels {0, 1} only; there is no multi-class or weighted averaging.
- The regression example from the task template (MAE) does not apply to this image classification item, so it is not implemented.
- No randomness, so no seed is needed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
