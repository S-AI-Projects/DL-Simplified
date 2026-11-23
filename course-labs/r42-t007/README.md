# R42-T007 · Lung CT (COVID) subset – CNN · Reproducible Configuration

**Student:** Rohit Kumar Rai  
**Task:** R42-T007 · 7 of 10  
**Branch:** `R42-T007`  
**Closes:** [S-AI-Projects/DL-Simplified#7](https://github.com/S-AI-Projects/DL-Simplified/issues/7)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images). It is **not** a lung CT dataset.
> All results are **pipeline verification**, not evidence of real-world or clinical performance.

---

## Goal

Write a serializable configuration for the image fixture covering seed, schema version, feature ordering, split IDs and
algorithm settings. Reloading it must reconstruct the same split and predictions, and unknown or contradictory settings
must be rejected.

## Configuration (`config.json`)

| Field | Value / rule |
|-------|--------------|
| `schema_version` | `"1.0"`; any other value is rejected |
| `seed` | `42`; drives the seeded group shuffle that made the split |
| `input_shape` | `[4, 4]` (rows, cols) |
| `feature_order` | row-major `p00, p01, …, p33`; must match `input_shape`, otherwise rejected as contradictory |
| `split` | exact record IDs: train `r002, r005` / validation `r001, r004` / test `r003, r006`; an ID in two splits is rejected |
| `algorithm` | `name: nearest_centroid`, `distance: squared_euclidean`, `pixel_scale: 255` (must be > 0), `tie_break: smallest_label` |
| `environment` | Python version and platform; recorded, not enforced |

**Rejected:**

- unknown top-level or algorithm fields;
- missing fields;
- an unsupported schema version or algorithm;
- `feature_order` that contradicts `input_shape`;
- overlapping split IDs, or a non-positive `pixel_scale`;
- at run time, input records with an incompatible shape (e.g. 9 pixels or a nested 4×4 list) and records not in the
  configured split.

## Data contract and fixture

| record_id | group_id | pixels | label | split |
|-----------|----------|--------|-------|-------|
| r001 | g1 | all-zero | 0 | validation |
| r004 | g1 | white diagonal (255) | 1 | validation |
| r002 | g2 | all-zero | 0 | train |
| r005 | g2 | white diagonal (255) | 1 | train |
| r003 | g3 | all-zero | 0 | test |
| r006 | g3 | white diagonal (255) | 1 | test |

Fields: `record_id` (str, unique), `group_id` (str), `pixels` (16 ints, 0–255, row-major 4×4), `label` (0/1).

## Result

- Running from the original config and from the saved-then-reloaded config both give the same split (above) and predictions
  `{r001: 0, r004: 1, r003: 0, r006: 1}` (nearest centroid fitted on train only).
- Regenerating the config with seed 42 gives `config.json` exactly. The `environment` block is the only part that may differ
  between machines.

---

## Files

```
course-labs/r42-t007/
├── fixture.json          ← literal synthetic fixture (simulated data)
├── config.json           ← saved configuration (seed 42)
├── expected_result.json  ← expected split + predictions
├── pipeline.py           ← make / validate / save / load config, run_from_config
├── test_pipeline.py      ← unit tests
└── README.md             ← this file
work/R42-T007/             ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json    ← input
├── config_written.json   ← config written during the evidence run
├── config_result.json    ← expected vs actual (original and reloaded)
├── tool_versions.json    ← Python / platform
├── unittest_output.txt   ← verbose test run (17 tests, exit_code=0)
└── mutation_check.txt    ← one expected prediction changed → FAILED, exit_code=1
```

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # no packages to install (standard library only)
python3 course-labs/r42-t007/pipeline.py              # writes work/R42-T007/ evidence
python3 -m unittest discover -s course-labs/r42-t007 -p "test_*.py"
```

## Done When

- [x] Reloaded configuration reconstructs the same split and predictions
- [x] An unknown field and an incompatible input shape are rejected
- [x] Environment versions are recorded
- [x] Fixture: six synthetic 4×4 images (3 all-zero → 0, 3 white diagonal → 1), unique record IDs, declared group IDs
- [x] Check is offline, CPU-only, bounded; reports a nonzero test count (17) and fails when an expected result is changed
- [x] Results labelled as pipeline verification only

## Limits

- The environment is recorded but not enforced, so a different Python version can still load the config.
- Only the nearest-centroid settings listed above are supported; a CNN would need its own algorithm block.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only) |
| Third-party packages | none |
