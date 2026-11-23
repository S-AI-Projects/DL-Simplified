# R42-T026 · Add a bounded CPU smoke run: News headlines sarcasm – embedding + LSTM

**Student:** Shashank Suyash  
**Enrollment:** `E26B07F0738`  
**Task:** R42-T026 · 6 of 10  
**Branch:** `R42-T026`  
**Closes:** [S-AI-Projects/DL-Simplified#26](https://github.com/S-AI-Projects/DL-Simplified/issues/26)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six fictional text records).  
> It is **not** the real dataset and must not be used to make performance claims about real-world tasks.

---

## Goal

Add a bounded CPU smoke run: News headlines sarcasm – embedding + LSTM

---

## Automated Check

```bash
python3 -m unittest discover -s course-labs/r42-t026 -p "test_*.py" -v
```

---

## Evidence (`work/`)

Contains:
- `fixture_input.json`: The synthetic fixture input
- `tool_versions.json`: Platform and Python version details
- `check_result.json`: Summary output of the test run
