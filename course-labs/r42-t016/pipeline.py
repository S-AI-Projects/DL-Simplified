"""Bounded CPU smoke pipeline using a deterministic synthetic-data baseline."""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).with_name("fixture.json")
SEED = 42
MAX_SAMPLES = 32
MAX_STEPS = 20


def load_fixture(path=FIXTURE):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    records = data["records"]
    if not records or len(records) > MAX_SAMPLES:
        raise ValueError("Fixture must contain 1 to 32 records")

    record_ids = [r["record_id"] for r in records]
    group_ids = [r["group_id"] for r in records]

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Duplicate record IDs")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("Duplicate group IDs")

    groups_by_split = {}
    for record in records:
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        groups_by_split.setdefault(record["group_id"], set()).add(
            record["split"]
        )
        if record["label"] not in (0, 1):
            raise ValueError("Invalid label")

        pixels = np.asarray(record["pixels"])
        if pixels.shape != (4, 4):
            raise ValueError("Each image must be 4x4")
        if not np.issubdtype(pixels.dtype, np.integer):
            raise ValueError("Pixels must be integers")
        if np.any((pixels < 0) | (pixels > 255)):
            raise ValueError("Pixels must be in range 0..255")

    if any(len(splits) > 1 for splits in groups_by_split.values()):
        raise ValueError("Group leakage across splits")

    if {r["split"] for r in records} != {"train", "validation", "test"}:
        raise ValueError("Train, validation and test splits are required")

    return data


def run_smoke(path=FIXTURE, seed=SEED):
    np.random.seed(seed)
    data = load_fixture(path)
    records = data["records"]

    # Deterministic baseline: any non-zero pixel predicts class 1.
    predictions = np.array([
        int(np.any(np.asarray(record["pixels"]) > 0))
        for record in records
    ], dtype=np.int64)

    labels = np.array([record["label"] for record in records])
    loss = float(np.mean((predictions - labels) ** 2))

    result = {
        "baseline": "deterministic_nonzero_pixel",
        "seed": seed,
        "sample_count": len(records),
        "prediction_shape": list(predictions.shape),
        "predictions": predictions.tolist(),
        "loss": loss,
        "step_count": 0,
        "device": "CPU",
    }

    assert np.isfinite(loss)
    assert predictions.shape == (len(records),)
    assert result["sample_count"] <= MAX_SAMPLES
    assert result["step_count"] <= MAX_STEPS
    return result


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
