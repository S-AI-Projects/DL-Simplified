
# R42-T012: Prevent Split Leakage

## Objective

Create a deterministic train, validation, and test split for a small image fixture while ensuring that records belonging to the same group never appear in different splits.

## Dataset

This task uses a synthetic CPU smoke fixture containing six simulated 4x4 grayscale images.

**Important:** This is simulated data, not a real cassava leaf dataset. It must not be used to make claims about real-world cassava disease classification performance.

Each record contains:

| Field | Description |
|---|---|
| `record_id` | Unique image record identifier |
| `group_id` | Group used to prevent data leakage |
| `pixels` | 4x4 grayscale pixel values (0–255) |
| `label` | Binary class label (0 or 1) |

The fixture contains three groups, each with two records.

No user IDs or timestamps are included in this synthetic fixture. For a real dataset, additional user or time-based isolation may be necessary.

## Implementation

`pipeline.py` provides:

- Input validation for required fields.
- Duplicate record ID detection.
- Image shape and pixel-range validation.
- Deterministic splitting using a fixed random seed.
- Group-based separation across train, validation, and test.
- A split manifest containing exact group IDs and record IDs.

The default seed is `42`.

The target split proportions are approximately 60% train, 20% validation, and 20% test by group count. With only three groups, each split receives one group.

## Running the Pipeline

From the repository root:

```bash
python3 course-labs/r42-t012/pipeline.py
```

## Running Tests

```bash
python3 -m unittest discover -s course-labs/r42-t012 -p "test_*.py" -v
```

The test suite checks:

1. Every record is assigned exactly once.
2. Groups do not overlap across splits.
3. The same seed produces the same manifest.
4. Duplicate record IDs are rejected.
5. Invalid pixel values are rejected.
6. Fewer than three groups are rejected.

## Environment

- Python: 3.14.6
- Dependencies: Python standard library only
- Execution: CPU-only
- External datasets: None
- Model training: Not performed

## Limitations

This implementation demonstrates leakage prevention on a small synthetic fixture. It does not train a transfer-learning model or evaluate classification accuracy.

For a real dataset, the grouping strategy must reflect the actual data collection process. Related images, users, or time-dependent records may require additional isolation rules.

