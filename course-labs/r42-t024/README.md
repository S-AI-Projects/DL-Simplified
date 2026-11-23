# R42-T024 · Verify the evaluation metric: GTSRB traffic signs – CNN

**Student:** Shashank Suyash  
**Enrollment:** `E26B07F0738`  
**Task:** R42-T024 · 4 of 10  
**Branch:** `R42-T024`  
**Closes:** [S-AI-Projects/DL-Simplified#24](https://github.com/S-AI-Projects/DL-Simplified/issues/24)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the real dataset and must not be used to make performance claims about real-world tasks.

---

## Goal

Verify the evaluation metric: GTSRB traffic signs – CNN

---

## Automated Check

```bash
python3 -m unittest discover -s course-labs/r42-t024 -p "test_*.py" -v
```

---

## Evidence (`work/`)

Contains:
- `fixture_input.json`: The synthetic fixture input
- `tool_versions.json`: Platform and Python version details
- `check_result.json`: Summary output of the test run
