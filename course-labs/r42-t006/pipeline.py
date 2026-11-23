"""
R42-T006 · Concrete crack detection – CNN
Task type : Small training/inference smoke test (bounded CPU run)
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture, NOT a concrete crack dataset. The run
      checks interface correctness only, not useful accuracy.

Smoke run
---------
Model   : tiny logistic regression, 16 inputs (pixels / 255) → 1 probability.
          It is a learned model, NOT a CNN and NOT a constant/naive model.
Budget  : at most 32 samples, exactly one epoch, at most 20 update steps
          (one SGD step per training sample), fixed seed. Exceeding any limit
          raises ValueError before training starts.
Split   : train g1 / validation g2 / test g3 (split.json). Only train
          records update the weights; validation loss and test predictions
          are computed after the epoch.
Needs   : Python standard library only — no GPU, credentials or downloads.
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import random
import sys
from typing import Dict, List, Optional

SEED = 0
LEARNING_RATE = 0.5
MAX_SAMPLES = 32
MAX_EPOCHS = 1
MAX_STEPS = 20
NUM_PIXELS = 16
SPLITS = ("train", "validation", "test")
MODEL_DESCRIPTION = "tiny logistic regression (16 -> 1), learned; not a CNN, not a constant/naive model"

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T006"


def load_json(name: str):
    return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return load_json("fixture.json")


def split_records(records: List[dict], split: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    """Assign records to splits by group_id; reject overlaps, unknown groups and empty splits."""
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id")
    owner: Dict[str, str] = {}
    for name in SPLITS:
        for gid in split[name]:
            if gid in owner:
                raise ValueError(f"group_id {gid!r} is in both {owner[gid]!r} and {name!r}")
            owner[gid] = name
    parts: Dict[str, List[dict]] = {name: [] for name in SPLITS}
    for rec in records:
        if rec["group_id"] not in owner:
            raise ValueError(f"record {rec['record_id']!r}: group {rec['group_id']!r} is in no split")
        parts[owner[rec["group_id"]]].append(rec)
    for name in SPLITS:
        if not parts[name]:
            raise ValueError(f"split {name!r} is empty")
        parts[name].sort(key=lambda r: r["record_id"])
    return parts


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------

def _features(rec: dict) -> List[float]:
    pixels = rec["pixels"]
    if len(pixels) != NUM_PIXELS:
        raise ValueError(f"record {rec['record_id']!r}: expected {NUM_PIXELS} pixels")
    return [p / 255.0 for p in pixels]


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _log_loss(y: int, p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


class TinyLogisticModel:
    def __init__(self, seed: int = SEED) -> None:
        rng = random.Random(seed)
        self.weights = [rng.uniform(-0.01, 0.01) for _ in range(NUM_PIXELS)]
        self.bias = 0.0

    def predict_proba(self, records: List[dict]) -> List[float]:
        return [
            _sigmoid(sum(w * x for w, x in zip(self.weights, _features(r))) + self.bias)
            for r in records
        ]

    def loss(self, records: List[dict]) -> float:
        probs = self.predict_proba(records)
        return sum(_log_loss(r["label"], p) for r, p in zip(records, probs)) / len(records)

    def sgd_step(self, rec: dict, lr: float) -> float:
        x = _features(rec)
        p = self.predict_proba([rec])[0]
        grad = p - rec["label"]                     # d(log loss)/dz
        self.weights = [w - lr * grad * xi for w, xi in zip(self.weights, x)]
        self.bias -= lr * grad
        return _log_loss(rec["label"], p)


def smoke_run(
    records: List[dict],
    split: Dict[str, List[str]],
    seed: int = SEED,
    epochs: int = 1,
    lr: float = LEARNING_RATE,
) -> dict:
    """Train for one bounded epoch on train, then evaluate validation loss and test predictions."""
    if len(records) > MAX_SAMPLES:
        raise ValueError(f"budget: {len(records)} samples > {MAX_SAMPLES}")
    if epochs > MAX_EPOCHS or epochs < 1:
        raise ValueError(f"budget: epochs must be 1..{MAX_EPOCHS}, got {epochs}")
    parts = split_records(records, split)
    planned_steps = epochs * len(parts["train"])
    if planned_steps > MAX_STEPS:
        raise ValueError(f"budget: would take {planned_steps} update steps > {MAX_STEPS}")

    model = TinyLogisticModel(seed)
    order = list(parts["train"])
    rng = random.Random(seed)
    step_losses: List[float] = []
    for _ in range(epochs):
        rng.shuffle(order)
        for rec in order:
            step_losses.append(model.sgd_step(rec, lr))

    test_probs = model.predict_proba(parts["test"])
    return {
        "model": MODEL_DESCRIPTION,
        "is_constant_model": False,
        "seed": seed,
        "learning_rate": lr,
        "epochs": epochs,
        "steps": len(step_losses),
        "train_order": [r["record_id"] for r in order],
        "step_losses": step_losses,
        "train_loss_after": model.loss(parts["train"]),
        "validation_loss": model.loss(parts["validation"]),
        "test_record_ids": [r["record_id"] for r in parts["test"]],
        "test_probabilities": test_probs,
        "test_prediction_shape": [len(test_probs)],
    }


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_json("split.json")
    actual = smoke_run(fixture, split)
    expected = load_json("expected_result.json")

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "split": split}, indent=2), encoding="utf-8"
    )
    result = {
        "note": (
            "Pipeline verification only (interface smoke test). Synthetic 4×4 "
            "fixture — NOT a concrete crack dataset; no accuracy claim."
        ),
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "third_party_packages": "none (standard library only)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[R42-T006] steps={actual['steps']} validation_loss={actual['validation_loss']:.6f} "
          f"test shape={actual['test_prediction_shape']} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
