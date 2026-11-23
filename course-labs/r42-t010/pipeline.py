"""
R42-T010 · Food-101 subset – CNN
Task type : Saved-model interface
Fixture   : Six synthetic 4×4 grayscale images (fixture.json)
            - three all-zero images  → label 0
            - three white-diagonal   → label 1
            Each record has a unique record_id and a declared group_id.
NOTE: This is a CPU smoke fixture, NOT Food-101. Results are pipeline
      verification only, not evidence of real-world performance.

Saved pipeline (JSON only — no pickle is ever written or loaded)
-----------------------------------------------------------------
{
  "format": "r42-t010-nearest-centroid",
  "format_version": 1,
  "schema": {"required_fields": ["record_id", "pixels"], "pixels_length": 16,
             "pixel_min": 0, "pixel_max": 255, "labels": [0, 1]},
  "params": {"pixel_scale": 255, "centroids": {"0": [16 floats], "1": [16 floats]}}
}

* `fit_pipeline` learns one centroid per label from TRAIN records only.
* `save_pipeline` / `load_pipeline` use `.json` files only; any other suffix
  (e.g. `.pkl`) is refused before the file is opened.
* `load_pipeline` fails clearly on a wrong format, a version mismatch, a schema
  mismatch or malformed parameters.
* `predict` validates every input record against the saved schema and
  returns {record_id, label, scores}; scores are squared Euclidean distances
  to each centroid (pixels / 255), tie → smallest label. Inference needs only
  the saved file and the input pixels: no training data, labels or credentials.
* Before/after-reload predictions must agree exactly on labels and within
  TOLERANCE = 1e-9 (absolute) on scores.
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
from typing import Dict, List, Optional

FORMAT = "r42-t010-nearest-centroid"
FORMAT_VERSION = 1
SCHEMA = {
    "required_fields": ["record_id", "pixels"],
    "pixels_length": 16,
    "pixel_min": 0,
    "pixel_max": 255,
    "labels": [0, 1],
}
PIXEL_SCALE = 255
TOLERANCE = 1e-9
SPLITS = ("train", "validation", "test")

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T010"


def load_json(name: str):
    return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return load_json("fixture.json")


def split_records(records: List[dict], split: Dict[str, List[str]]) -> Dict[str, List[dict]]:
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
        parts[owner[rec["group_id"]]].append(rec)
    for name in SPLITS:
        parts[name].sort(key=lambda r: r["record_id"])
    return parts


# ---------------------------------------------------------------------------
# Schema validation of inference input
# ---------------------------------------------------------------------------

def validate_input(rec: dict, schema: dict) -> List[float]:
    if not isinstance(rec, dict):
        raise ValueError("input record must be an object")
    for field in schema["required_fields"]:
        if field not in rec:
            raise ValueError(f"input schema: missing field {field!r}")
    rid = rec["record_id"]
    if not isinstance(rid, str) or not rid:
        raise ValueError("input schema: record_id must be a non-empty string")
    pixels = rec["pixels"]
    if not isinstance(pixels, list) or len(pixels) != schema["pixels_length"]:
        got = len(pixels) if isinstance(pixels, list) else type(pixels).__name__
        raise ValueError(f"input schema: record {rid!r} needs {schema['pixels_length']} pixels, got {got}")
    for p in pixels:
        if isinstance(p, bool) or not isinstance(p, int) or not schema["pixel_min"] <= p <= schema["pixel_max"]:
            raise ValueError(
                f"input schema: record {rid!r} pixel {p!r} is not an int in "
                f"[{schema['pixel_min']}, {schema['pixel_max']}]"
            )
    return [p / PIXEL_SCALE for p in pixels]


# ---------------------------------------------------------------------------
# Fit / save / load / predict
# ---------------------------------------------------------------------------

def fit_pipeline(train_records: List[dict]) -> dict:
    """One centroid per label from training records only."""
    centroids = {}
    for label in SCHEMA["labels"]:
        vecs = [validate_input(r, SCHEMA) for r in train_records if r["label"] == label]
        if not vecs:
            raise ValueError(f"no training records for label {label}")
        centroids[str(label)] = [sum(col) / len(vecs) for col in zip(*vecs)]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "schema": dict(SCHEMA),
        "params": {"pixel_scale": PIXEL_SCALE, "centroids": centroids},
    }


def _require_json_path(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.suffix != ".json":
        raise ValueError(f"refusing {path.name!r}: only .json pipelines are supported (never pickle)")
    return path


def save_pipeline(pipeline: dict, path: pathlib.Path) -> None:
    check_pipeline(pipeline)
    _require_json_path(path).write_text(json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_pipeline(path: pathlib.Path) -> dict:
    pipeline = json.loads(_require_json_path(path).read_text(encoding="utf-8"))
    check_pipeline(pipeline)
    return pipeline


def check_pipeline(pipeline: dict) -> None:
    if not isinstance(pipeline, dict) or pipeline.get("format") != FORMAT:
        raise ValueError(f"format mismatch: expected {FORMAT!r}, got {pipeline.get('format') if isinstance(pipeline, dict) else None!r}")
    if pipeline.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"version mismatch: file has format_version {pipeline.get('format_version')!r}, "
            f"this code supports {FORMAT_VERSION}"
        )
    if pipeline.get("schema") != SCHEMA:
        raise ValueError(f"schema mismatch: saved schema {pipeline.get('schema')!r} != expected {SCHEMA!r}")
    params = pipeline.get("params", {})
    if params.get("pixel_scale") != PIXEL_SCALE:
        raise ValueError("params: pixel_scale mismatch")
    centroids = params.get("centroids", {})
    if sorted(centroids) != [str(l) for l in SCHEMA["labels"]]:
        raise ValueError("params: centroids must have one entry per label")
    for label, vec in centroids.items():
        if (not isinstance(vec, list) or len(vec) != SCHEMA["pixels_length"]
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec)):
            raise ValueError(f"params: centroid {label!r} must be {SCHEMA['pixels_length']} finite numbers")


def predict(pipeline: dict, records: List[dict]) -> List[dict]:
    """Schema-validated prediction; needs only the saved pipeline and input pixels."""
    check_pipeline(pipeline)
    centroids = {int(k): v for k, v in pipeline["params"]["centroids"].items()}
    out = []
    for rec in records:
        x = validate_input(rec, pipeline["schema"])
        scores = {lab: sum((a - b) ** 2 for a, b in zip(x, c)) for lab, c in centroids.items()}
        best = min(scores.values())
        out.append({
            "record_id": rec["record_id"],
            "label": min(lab for lab, s in scores.items() if s == best),
            "scores": {str(lab): s for lab, s in sorted(scores.items())},
        })
    return out


def predictions_agree(before: List[dict], after: List[dict], tol: float = TOLERANCE) -> bool:
    """Labels identical; every score within tol (absolute)."""
    if [p["record_id"] for p in before] != [p["record_id"] for p in after]:
        return False
    for b, a in zip(before, after):
        if b["label"] != a["label"] or b["scores"].keys() != a["scores"].keys():
            return False
        if any(abs(b["scores"][k] - a["scores"][k]) > tol for k in b["scores"]):
            return False
    return True


def inference_inputs(records: List[dict]) -> List[dict]:
    """Strip everything except the schema's required fields (no labels at inference)."""
    return [{k: r[k] for k in SCHEMA["required_fields"]} for r in records]


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_json("split.json")
    parts = split_records(fixture, split)
    pipe = fit_pipeline(parts["train"])
    held_out = inference_inputs(parts["validation"] + parts["test"])
    before = predict(pipe, held_out)
    save_pipeline(pipe, out / "saved_pipeline.json")
    after = predict(load_pipeline(out / "saved_pipeline.json"), held_out)

    actual = {"before_reload": before, "after_reload": after,
              "agree_within_tolerance": predictions_agree(before, after), "tolerance": TOLERANCE}
    expected = load_json("expected_result.json")

    (out / "fixture_input.json").write_text(json.dumps({"records": fixture, "split": split}, indent=2),
                                            encoding="utf-8")
    result = {
        "note": "Pipeline verification only. Synthetic 4×4 fixture — NOT Food-101.",
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "saved_pipeline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tool_versions.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "third_party_packages": "none (standard library only)"}, indent=2),
        encoding="utf-8",
    )
    print(f"[R42-T010] labels={[p['label'] for p in after]} agree={actual['agree_within_tolerance']} "
          f"outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
