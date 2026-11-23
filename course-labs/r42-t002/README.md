# R42-T002 · NEU Steel Surface Defects – CNN · Leakage-Free Splitting

**Student:** Rohit Kumar Rai  
**Task:** R42-T002 · 2 of 10  
**Branch:** `R42-T002`  
**Closes:** [S-AI-Projects/DL-Simplified#2](https://github.com/S-AI-Projects/DL-Simplified/issues/2)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the NEU Steel Surface Defects dataset and must not be used to make performance claims about that dataset or any real-world defect detection task.

---

## Goal

Implement a **deterministic split manifest** for the image fixture that:
- Preserves **group isolation** (all records with the same `group_id` go to the same split)
- Writes the **exact record IDs** to each split
- Produces **identical output on every run** (no randomness)
- Rejects **intentionally overlapping IDs/groups**

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

## Split Manifest (Deterministic)

| Split | Groups | Record IDs |
|-------|--------|-----------|
| train | g1 | r001, r002 |
| val | g2 | r003, r004 |
| test | g3 | r005, r006 |

Groups are sorted (g1 < g2 < g3) and assigned to splits in order. This assignment is **frozen** in the code — no randomness is used.

---

## Files

```
course-labs/r42-t002/
├── pipeline.py        ← split manifest builder + validator
├── test_pipeline.py   ← unit tests
└── README.md          ← this file
work/R42-T002/
├── fixture_input.json
├── split_result.json  ← manifest + fingerprint + errors + outcome
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

# 3. Run the splitter (generates work/R42-T002/ evidence)
python3 course-labs/r42-t002/pipeline.py

# 4. Run the automated checks
python3 -m unittest discover -s course-labs/r42-t002 -p "test_*.py" -v
```

---

## Done When

- [x] Train / val / test record sets are **disjoint**
- [x] Repeated runs produce the **same manifest** (verified via SHA-256 fingerprint)
- [x] Intentionally overlapping IDs/groups are **rejected**
- [x] Group isolation is preserved
- [x] Checks are offline, CPU-only, bounded
- [x] Results labelled as pipeline verification only

---

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.x (stdlib) |
| NumPy   | ≥ 1.21 (fixture generation only) |
