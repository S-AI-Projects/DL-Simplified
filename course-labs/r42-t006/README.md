# R42-T006 · Concrete crack detection – CNN · Bounded CPU Smoke Run

**Student:** Rohit Kumar Rai  
**Task:** R42-T006 · 6 of 10  
**Branch:** `R42-T006`  
**Closes:** [S-AI-Projects/DL-Simplified#6](https://github.com/S-AI-Projects/DL-Simplified/issues/6)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** a concrete crack dataset.
> The run checks **interface correctness only**. Results are **pipeline verification**, not useful accuracy.

---

## Goal

A tiny image pipeline smoke test: at most 32 samples, a fixed seed, and at most one training epoch / 20 update steps, on CPU.
It asserts finite outputs and loss, the expected prediction shape, and a recorded step count.

## Model (explicitly labelled)

A **tiny logistic regression**: 16 inputs (pixels / 255) → 1 probability, trained with SGD.

- It is a learned model: **not a CNN** and **not a constant/naive model**. The result records
  `"is_constant_model": false` and the model description.
- Weights are initialised from `random.Random(0)` in [−0.01, 0.01]. Bias is 0, learning rate 0.5.

## Budget (enforced before training)

| Limit | Value | On violation |
|-------|-------|--------------|
| samples | ≤ 32 | `ValueError` |
| epochs | exactly 1 | `ValueError` |
| update steps | ≤ 20 (one per training sample) | `ValueError` |

No GPU, credentials or dataset download are needed. The run only reads local JSON files with the Python standard library.

## Data contract and split

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | unique |
| `group_id` | `str` | assigns the record to a split (`split.json`) |
| `pixels` | `list[int]` | 16 values (4×4, row-major), 0–255 |
| `label` | `int` | `0` or `1` |

| Split | Groups | Records |
|-------|--------|---------|
| train | g1 | r001 (all-zero, 0), r004 (white diagonal, 1) |
| validation | g2 | r002 (0), r005 (1) |
| test | g3 | r003 (0), r006 (1) |

Only train records update the weights. Validation loss and test predictions are computed after the epoch.

## Result (seed 0)

| Item | Value |
|------|-------|
| epochs / steps | 1 / **2** |
| step losses | 0.693147, 0.821361 |
| train loss after epoch | 0.490663 |
| validation loss | 0.490663 |
| test prediction shape | **[2]** |
| test probabilities (r003, r006) | 0.507520, 0.761074 |

All values are finite, and repeated runs are identical. Full values are in `expected_result.json`.
No accuracy is claimed.

---

## Files

```
course-labs/r42-t006/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── split.json            ← fixed group-level train/validation/test split
├── expected_result.json  ← expected smoke-run output for seed 0
├── pipeline.py           ← bounded training + inference
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T006/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json    ← input records + split
├── smoke_result.json     ← expected vs actual
├── tool_versions.json    ← Python / platform
├── unittest_output.txt   ← verbose test run (13 tests, exit_code=0)
└── mutation_check.txt    ← expected step count changed 2 → 3 → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t006/pipeline.py              # writes work/R42-T006/ evidence
python3 -m unittest discover -s course-labs/r42-t006 -p "test_*.py"
```

## Done When

- [x] Finite outputs and loss, expected prediction shape, recorded step count
- [x] No GPU, credentials or dataset download needed
- [x] Model type explicitly labelled (learned logistic regression, not constant/naive)
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (13) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- Not a CNN. A logistic regression is used to keep the run standard-library-only and bounded. The interface (fit → finite
  loss → predictions of the right shape) is what is tested.
- Two training samples and two update steps: the numbers carry no information about crack detection.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
