"""
R42-T007 · Lung CT (COVID) subset – CNN
Task type : Configuration reproducibility
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture, NOT a lung CT dataset. Results are
      pipeline verification only, not evidence of clinical performance.

Configuration (config.json)
---------------------------
schema_version : "1.0"
seed           : int, used for the seeded group shuffle that makes the split
input_shape    : [4, 4] (rows, cols)
feature_order  : row-major pixel names "p00".."p33" (must match input_shape)
split          : exact record IDs for train / validation / test
algorithm      : nearest-centroid settings
                 {name: "nearest_centroid", distance: "squared_euclidean",
                  pixel_scale: 255, tie_break: "smallest_label"}
environment    : Python version and platform (recorded, not enforced)

`load_config` rejects unknown fields, missing fields, an unsupported schema
version or algorithm, and contradictory settings (feature_order vs
input_shape, overlapping split IDs, non-positive pixel_scale).
`run_from_config` rejects records whose pixel count does not match
input_shape, and records that are not in the configured split.
"""

from __future__ import annotations

import json
import pathlib
import platform
import random
import sys
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0"
SPLITS = ("train", "validation", "test")
TOP_LEVEL_FIELDS = {"schema_version", "seed", "input_shape", "feature_order",
                    "split", "algorithm", "environment"}
ALGORITHM_FIELDS = {"name", "distance", "pixel_scale", "tie_break"}
ALGORITHM_DEFAULTS = {
    "name": "nearest_centroid",
    "distance": "squared_euclidean",
    "pixel_scale": 255,
    "tie_break": "smallest_label",
}
DEFAULT_SEED = 42
INPUT_SHAPE = [4, 4]

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T007"


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return json.loads((LAB_DIR / "fixture.json").read_text(encoding="utf-8"))


def feature_names(shape: List[int]) -> List[str]:
    rows, cols = shape
    return [f"p{r}{c}" for r in range(rows) for c in range(cols)]


def environment() -> Dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}


# ---------------------------------------------------------------------------
# Build / save / load configuration
# ---------------------------------------------------------------------------

def make_config(records: List[dict], seed: int = DEFAULT_SEED) -> dict:
    """Seeded group shuffle → one whole group per split; record the exact IDs."""
    groups: Dict[str, List[str]] = {}
    for rec in records:
        groups.setdefault(rec["group_id"], []).append(rec["record_id"])
    if len(groups) != len(SPLITS):
        raise ValueError(f"expected {len(SPLITS)} groups, got {len(groups)}")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input_shape": list(INPUT_SHAPE),
        "feature_order": feature_names(INPUT_SHAPE),
        "split": {name: sorted(groups[g]) for name, g in zip(SPLITS, order)},
        "algorithm": dict(ALGORITHM_DEFAULTS),
        "environment": environment(),
    }
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    unknown = sorted(set(config) - TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown config field(s): {unknown}")
    missing = sorted(TOP_LEVEL_FIELDS - set(config))
    if missing:
        raise ValueError(f"missing config field(s): {missing}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {config['schema_version']!r}, expected {SCHEMA_VERSION!r}")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("seed must be an int")

    shape = config["input_shape"]
    if (not isinstance(shape, list) or len(shape) != 2
            or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape)):
        raise ValueError(f"input_shape must be [rows, cols] of positive ints, got {shape!r}")
    if config["feature_order"] != feature_names(shape):
        raise ValueError("contradictory settings: feature_order does not match row-major input_shape")

    algo = config["algorithm"]
    unknown_algo = sorted(set(algo) - ALGORITHM_FIELDS)
    if unknown_algo:
        raise ValueError(f"unknown algorithm field(s): {unknown_algo}")
    missing_algo = sorted(ALGORITHM_FIELDS - set(algo))
    if missing_algo:
        raise ValueError(f"missing algorithm field(s): {missing_algo}")
    for key in ("name", "distance", "tie_break"):
        if algo[key] != ALGORITHM_DEFAULTS[key]:
            raise ValueError(f"unsupported algorithm {key} {algo[key]!r}")
    scale = algo["pixel_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"contradictory settings: pixel_scale must be > 0, got {scale!r}")

    split = config["split"]
    if set(split) != set(SPLITS):
        raise ValueError(f"split must have exactly {list(SPLITS)}")
    seen: Dict[str, str] = {}
    for name in SPLITS:
        if not split[name]:
            raise ValueError(f"split {name!r} is empty")
        for rid in split[name]:
            if rid in seen:
                raise ValueError(f"contradictory settings: record_id {rid!r} is in both {seen[rid]!r} and {name!r}")
            seen[rid] = name


def config_to_json(config: dict) -> str:
    return json.dumps(config, indent=2, sort_keys=True) + "\n"


def save_config(config: dict, path: pathlib.Path) -> None:
    validate_config(config)
    pathlib.Path(path).write_text(config_to_json(config), encoding="utf-8")


def load_config(path: pathlib.Path) -> dict:
    config = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


# ---------------------------------------------------------------------------
# Reconstruct split and predictions from a configuration
# ---------------------------------------------------------------------------

def _features(rec: dict, config: dict) -> List[float]:
    rows, cols = config["input_shape"]
    pixels = rec["pixels"]
    if not isinstance(pixels, list) or len(pixels) != rows * cols or any(isinstance(p, list) for p in pixels):
        raise ValueError(
            f"record {rec['record_id']!r}: incompatible input shape, expected {rows * cols} "
            f"flat pixels for input_shape {config['input_shape']}"
        )
    scale = config["algorithm"]["pixel_scale"]
    return [p / scale for p in pixels]


def run_from_config(config: dict, records: List[dict]) -> dict:
    """Rebuild the split from config IDs, fit nearest centroids on train, predict held-out."""
    validate_config(config)
    by_id = {r["record_id"]: r for r in records}
    configured = {rid for name in SPLITS for rid in config["split"][name]}
    unknown = sorted(configured - set(by_id))
    extra = sorted(set(by_id) - configured)
    if unknown or extra:
        raise ValueError(f"records do not match configured split: unknown={unknown} unassigned={extra}")

    train = [by_id[rid] for rid in config["split"]["train"]]
    centroids: Dict[int, List[float]] = {}
    for label in sorted({r["label"] for r in train}):
        vecs = [_features(r, config) for r in train if r["label"] == label]
        centroids[label] = [sum(col) / len(vecs) for col in zip(*vecs)]

    def predict(rec: dict) -> int:
        x = _features(rec, config)
        dists = {lab: sum((a - b) ** 2 for a, b in zip(x, c)) for lab, c in centroids.items()}
        best = min(dists.values())
        return min(lab for lab, d in dists.items() if d == best)   # tie → smallest label

    return {
        "split": {name: list(config["split"][name]) for name in SPLITS},
        "predictions": {
            rid: predict(by_id[rid])
            for name in ("validation", "test") for rid in config["split"][name]
        },
    }


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture = load_fixture()
    config = make_config(fixture)
    save_config(config, out / "config_written.json")
    reloaded = load_config(out / "config_written.json")
    actual = {
        "original": run_from_config(config, fixture),
        "reloaded": run_from_config(reloaded, fixture),
    }
    expected = json.loads((LAB_DIR / "expected_result.json").read_text(encoding="utf-8"))

    (out / "fixture_input.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    result = {
        "note": "Pipeline verification only. Synthetic 4×4 fixture — NOT a lung CT dataset.",
        "expected": {"original": expected, "reloaded": expected},
        "actual": actual,
        "outcome": "PASS" if actual["original"] == expected == actual["reloaded"] else "FAIL",
    }
    (out / "config_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "third_party_packages": "none (standard library only)"}, indent=2),
        encoding="utf-8",
    )
    print(f"[R42-T007] split={actual['reloaded']['split']} "
          f"predictions={actual['reloaded']['predictions']} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
