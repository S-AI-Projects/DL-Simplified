# R42-T021 · Validate the data contract: DCGAN – MNIST digits

**Student:** Shashank Suyash  
**Enrollment:** `E26B07F0738`  
**Task:** R42-T021 · 1 of 10  
**Branch:** `R42-T021`  
**Closes:** [S-AI-Projects/DL-Simplified#21](https://github.com/S-AI-Projects/DL-Simplified/issues/21)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the real dataset and must not be used to make performance claims about real-world tasks.

---

## Goal

Validate the data contract: DCGAN – MNIST digits

---

## Automated Check

```bash
python3 -m unittest discover -s course-labs/r42-t021 -p "test_*.py" -v
```

---

## Evidence (`work/`)

Contains:
- `fixture_input.json`: The synthetic fixture input
- `tool_versions.json`: Platform and Python version details
- `check_result.json`: Summary output of the test run
