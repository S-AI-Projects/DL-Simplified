"""
R42-T005 · UrbanSound8K – audio classification
Task type : Preprocessing contract
Fixture   : Six synthetic eight-sample mono waveforms (fixture.json)
            - three all-zero arrays                      → label 0
            - three alternating [-1,1,-1,1,-1,1,-1,1]   → label 1
            sample_rate = 8000, distinct record IDs, declared group IDs.
NOTE: These are shape/processing fixtures, not meaningful sound classes and
      NOT UrbanSound8K. Results are pipeline verification only.

Preprocessing contract
----------------------
Input  : record with sample_rate (int, must equal 8000) and samples
         (flat list of exactly 8 int/float values = one mono channel).
         A mismatched rate, wrong length or multi-channel/nested input is
         rejected — never resampled, padded or truncated silently.
Output : float64 array of shape (channels, time) = (1, 8), channel-first,
         time order preserved.
Fitted : per-dataset standardisation x → (x - mean) / std, where mean and
         std (population) are computed over all TRAIN samples only.

Leakage guard
-------------
`WaveformPreprocessor.fit` only accepts a `SplitDataset` and reads its train
records; there is no public way to pass held-out records to `fit`. Building a
`SplitDataset` rejects splits whose groups or record IDs overlap (e.g. the
deliberate `contaminated_split.json`).
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
from typing import Dict, List, Optional, Tuple

SAMPLE_RATE = 8000
NUM_SAMPLES = 8
NUM_CHANNELS = 1
SPLITS = ("train", "validation", "test")
OUTPUT_DTYPE = "float64"      # Python float

LAB_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
EVIDENCE_DIR = REPO_ROOT / "work" / "R42-T005"


# ---------------------------------------------------------------------------
# Fixture and split
# ---------------------------------------------------------------------------

def load_json(name: str):
    return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))


def load_fixture() -> List[dict]:
    """Load the literal six-record synthetic fixture (simulated data)."""
    return load_json("fixture.json")


class SplitDataset:
    """Records assigned to train / validation / test by group_id."""

    def __init__(self, records: List[dict], split: Dict[str, List[str]]):
        ids = [r["record_id"] for r in records]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"contamination: duplicate record_id(s) {dupes}")

        owner: Dict[str, str] = {}
        for name in SPLITS:
            for gid in split[name]:
                if gid in owner:
                    raise ValueError(
                        f"contamination: group_id {gid!r} is in both {owner[gid]!r} and {name!r}"
                    )
                owner[gid] = name

        self._parts: Dict[str, List[dict]] = {name: [] for name in SPLITS}
        for rec in records:
            name = owner.get(rec["group_id"])
            if name is None:
                raise ValueError(f"record {rec['record_id']!r}: group {rec['group_id']!r} is in no split")
            self._parts[name].append(rec)
        for name in SPLITS:
            if not self._parts[name]:
                raise ValueError(f"split {name!r} is empty")

    def records(self, name: str) -> List[dict]:
        return list(self._parts[name])

    def record_ids(self, name: str) -> List[str]:
        return sorted(r["record_id"] for r in self._parts[name])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def check_waveform(record: dict) -> List[float]:
    """Validate one record against the input contract; return its samples."""
    rid = record.get("record_id")
    rate = record.get("sample_rate")
    if isinstance(rate, bool) or rate != SAMPLE_RATE:
        raise ValueError(f"record {rid!r}: sample_rate mismatch, expected {SAMPLE_RATE}, got {rate!r}")
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"record {rid!r}: samples must be a list")
    if any(isinstance(s, list) for s in samples):
        raise ValueError(f"record {rid!r}: expected mono 1-D samples, got nested (multi-channel) input")
    if len(samples) != NUM_SAMPLES:
        raise ValueError(f"record {rid!r}: expected {NUM_SAMPLES} samples, got {len(samples)}")
    for s in samples:
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            raise ValueError(f"record {rid!r}: sample {s!r} is not a finite number")
    return [float(s) for s in samples]


def shape_of(array: List[List[float]]) -> Tuple[int, int]:
    return (len(array), len(array[0]) if array else 0)


class WaveformPreprocessor:
    """Validate, standardise with train-only statistics, return (1, 8) arrays."""

    def __init__(self) -> None:
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.fitted_on: List[str] = []

    def fit(self, dataset: SplitDataset) -> "WaveformPreprocessor":
        if not isinstance(dataset, SplitDataset):
            raise TypeError("fit() only accepts a SplitDataset; it reads the train split only")
        train = dataset.records("train")
        values = [v for rec in train for v in check_waveform(rec)]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std == 0.0:
            raise ValueError("training samples have zero variance; cannot standardise")
        self.mean, self.std = mean, std
        self.fitted_on = dataset.record_ids("train")
        return self

    def transform(self, record: dict) -> List[List[float]]:
        if self.mean is None:
            raise RuntimeError("preprocessor is not fitted")
        samples = check_waveform(record)
        return [[(v - self.mean) / self.std for v in samples]]   # (channels=1, time=8)


def run_preprocessing(records: List[dict], split: Dict[str, List[str]]) -> dict:
    dataset = SplitDataset(records, split)
    pre = WaveformPreprocessor().fit(dataset)
    outputs = {}
    for name in SPLITS:
        outputs[name] = {
            r["record_id"]: pre.transform(r)
            for r in sorted(dataset.records(name), key=lambda r: r["record_id"])
        }
    return {
        "fitted_on": pre.fitted_on,
        "train_mean": pre.mean,
        "train_std": pre.std,
        "output_shape": [NUM_CHANNELS, NUM_SAMPLES],
        "output_layout": "(channels, time)",
        "output_dtype": OUTPUT_DTYPE,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Entry point (evidence generation)
# ---------------------------------------------------------------------------

def run_and_save(output_dir: Optional[str] = None) -> dict:
    out = pathlib.Path(output_dir) if output_dir else EVIDENCE_DIR
    out.mkdir(parents=True, exist_ok=True)

    fixture, split = load_fixture(), load_json("split.json")
    actual = run_preprocessing(fixture, split)
    try:
        SplitDataset(fixture, load_json("contaminated_split.json"))
        contamination = "ACCEPTED (unexpected)"
    except ValueError as exc:
        contamination = f"rejected: {exc}"
    actual["contaminated_split"] = contamination
    expected = load_json("expected_result.json")

    (out / "fixture_input.json").write_text(
        json.dumps({"records": fixture, "split": split,
                    "contaminated_split": load_json("contaminated_split.json")}, indent=2),
        encoding="utf-8",
    )
    result = {
        "note": (
            "Pipeline verification only. Synthetic 8-sample waveforms — not "
            "meaningful sound classes and NOT UrbanSound8K."
        ),
        "expected": expected,
        "actual": actual,
        "outcome": "PASS" if actual == expected else "FAIL",
    }
    (out / "preprocessing_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    print(f"[R42-T005] mean={actual['train_mean']} std={actual['train_std']} "
          f"shape={actual['output_shape']} contamination {contamination} outcome: {result['outcome']}")
    return result


if __name__ == "__main__":
    run_and_save()
