# R42-T001 · Fashion-MNIST – CNN · Dataset Contract (Schema Validator)

**Student:** Rohit Kumar Rai  
**Task:** R42-T001 · 1 of 10  
**Branch:** `R42-T001`  
**Closes:** [S-AI-Projects/DL-Simplified#1](https://github.com/S-AI-Projects/DL-Simplified/issues/1)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the Fashion-MNIST dataset and must not be used to make performance claims about Fashion-MNIST or any real-world task.

---

## Goal

Implement a schema validator for image input records with fields:

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | non-empty, unique |
| `group_id` | `str` | non-empty |
| `pixels` | `list[int]` | length == 16 (4×4), values in \[0, 255\] |
| `label` | `int` | in `{0, 1}` |

Failures are **always reported** (record ID + field + reason) — records are never silently dropped.

---

## Synthetic Fixture

| record_id | group_id | pixels | label |
|-----------|----------|--------|-------|
| r001 | g1 | all-zero (4×4) | 0 |
| r002 | g1 | all-zero (4×4) | 0 |
| r003 | g2 | all-zero (4×4) | 0 |
| r004 | g2 | white diagonal (4×4) | 1 |
| r005 | g3 | white diagonal (4×4) | 1 |
| r006 | g3 | white diagonal (4×4) | 1 |

---

## Files

```
course-labs/r42-t001/
├── pipeline.py        ← schema validator
├── test_pipeline.py   ← unit tests
└── README.md          ← this file
work/R42-T001/
├── fixture_input.json
├── validation_result.json
└── tool_versions.json
```

---

## How to Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install numpy

# 3. Run the validator (generates work/R42-T001/ evidence)
python3 course-labs/r42-t001/pipeline.py

# 4. Run the automated checks
python3 -m unittest discover -s course-labs/r42-t001 -p "test_*.py" -v
```

---

## Done When

- [x] Valid fixture passes with zero failures
- [x] Missing field → reported with field name + reason
- [x] Malformed numeric / text / tensor data → reported
- [x] Duplicate `record_id` → reported
- [x] Checks are offline, CPU-only, bounded
- [x] Results labelled as pipeline verification only

---

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.x (stdlib) |
| NumPy   | ≥ 1.21 (fixture generation only) |
