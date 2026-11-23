# R42-T005 · UrbanSound8K – audio classification · Preprocessing Contract

**Student:** Rohit Kumar Rai  
**Task:** R42-T005 · 5 of 10  
**Branch:** `R42-T005`  
**Closes:** [S-AI-Projects/DL-Simplified#5](https://github.com/S-AI-Projects/DL-Simplified/issues/5)

---

## ⚠️ Data Notice

> This component uses **six synthetic eight-sample waveforms**. They are shape/processing fixtures, **not meaningful sound
> classes** and **not UrbanSound8K**. All results are **pipeline verification**, not real-world performance.

---

## Goal

Make preprocessing explicit:

- fixed-length samples with an explicit sample rate;
- channel/time axis conventions preserved;
- mismatched rates rejected;
- the fitted step learns from training data only.

## Data contract

**Input record**

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | unique |
| `group_id` | `str` | assigns the record to a split (`split.json`) |
| `sample_rate` | `int` | must be exactly **8000** — otherwise rejected (never resampled) |
| `samples` | `list[number]` | one mono channel, exactly **8** finite values — wrong length is rejected (never padded/truncated); nested/multi-channel input is rejected |
| `label` | `int` | `0` or `1` (carried through, not used by preprocessing) |

**Output** (per record)

- dtype: `float64` (Python `float`).
- Shape `(channels, time) = (1, 8)`, channel-first, with time order preserved.
- Values `x → (x − mean) / std`. Mean and population std are computed over all **train** samples only.

## Synthetic fixture and split

| record_id | group_id | sample_rate | samples | label | split |
|-----------|----------|-------------|---------|-------|-------|
| a001 | g1 | 8000 | [0,0,0,0,0,0,0,0] | 0 | train |
| a004 | g1 | 8000 | [-1,1,-1,1,-1,1,-1,1] | 1 | train |
| a002 | g2 | 8000 | [0,0,0,0,0,0,0,0] | 0 | validation |
| a005 | g2 | 8000 | [-1,1,-1,1,-1,1,-1,1] | 1 | validation |
| a003 | g3 | 8000 | [0,0,0,0,0,0,0,0] | 0 | test |
| a006 | g3 | 8000 | [-1,1,-1,1,-1,1,-1,1] | 1 | test |

## Leakage guard

- `WaveformPreprocessor.fit()` accepts only a `SplitDataset` and reads its train records. Passing a list of records raises
  `TypeError`, so held-out input cannot be used for fitting through the public interface.
- Building a `SplitDataset` rejects a group in two splits, or a duplicate record ID.
- `contaminated_split.json` is the deliberate contamination fixture: it puts group `g2` in both train and validation, and is
  rejected with `contamination: group_id 'g2' is in both 'train' and 'validation'`.

## Result

| Item | Value |
|------|-------|
| Fitted on | a001, a004 (train only) |
| Train mean / std | 0.0 / √0.5 ≈ 0.70711 |
| All-zero waveform → | `[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]` |
| Alternating waveform → | `[[-1.41421, 1.41421, -1.41421, 1.41421, -1.41421, 1.41421, -1.41421, 1.41421]]` (= ±√2) |
| Output shape / dtype | (1, 8) / float64 |
| Contamination fixture | rejected |

Changing validation/test samples does not change the fitted mean/std (tested). Full values: `expected_result.json`.

---

## Files

```
course-labs/r42-t005/
├── fixture.json             ← literal synthetic waveforms (simulated data)
├── split.json               ← fixed group-level train/validation/test split
├── contaminated_split.json  ← deliberate contamination fixture (must be rejected)
├── expected_result.json     ← expected outputs
├── pipeline.py              ← input checks, SplitDataset, WaveformPreprocessor
├── test_pipeline.py         ← unit tests
└── README.md                ← this file
work/R42-T005/                ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json       ← input records + splits
├── preprocessing_result.json← expected vs actual
├── tool_versions.json       ← Python / platform
├── unittest_output.txt      ← verbose test run (17 tests, exit_code=0)
└── mutation_check.txt       ← expected fitted record IDs changed → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t005/pipeline.py              # writes work/R42-T005/ evidence
python3 -m unittest discover -s course-labs/r42-t005 -p "test_*.py"
```

## Done When

- [x] The fixture's output values, dtype and shape are asserted
- [x] Fitting on held-out input is impossible through the public transform interface
- [x] A deliberate contamination fixture is rejected
- [x] Fixture: six synthetic eight-sample waveforms (3 all-zero → 0, 3 alternating → 1), sample_rate = 8000, distinct record IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (17) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- Mono, 8 samples, 8000 Hz only; there is no resampling, padding, framing or spectrogram step.
- Training data with zero variance is rejected, because standardisation would divide by zero.
- No randomness, so no seed is needed.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
