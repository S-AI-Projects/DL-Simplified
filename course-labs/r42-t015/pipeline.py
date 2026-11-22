"""Explicit, stateless preprocessing for synthetic grayscale image records."""

import json
from pathlib import Path

import numpy as np


IMAGE_SHAPE = (4, 4)
PIXEL_MIN = 0
PIXEL_MAX = 255
VALID_SPLITS = {"train", "validation", "test"}
REQUIRED_FIELDS = {"record_id", "group_id", "split", "pixels", "label"}


def preprocess_image(pixels):
    """Validate a 4x4 grayscale image and return float32 data shaped (4, 4, 1)."""
    try:
        image = np.asarray(pixels, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixels must form a rectangular 4x4 image") from exc

    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"image must have shape {IMAGE_SHAPE}, got {image.shape}")

    values = []
    for value in image.flat:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean pixel values are not allowed")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("pixel values must be numeric")
        if not np.isfinite(value):
            raise ValueError("pixel values must be finite")
        if value < PIXEL_MIN or value > PIXEL_MAX:
            raise ValueError("pixel values must be between 0 and 255")
        values.append(value)

    image = np.asarray(values, dtype=np.float32).reshape(IMAGE_SHAPE)
    return (image / np.float32(255.0))[..., np.newaxis]


def validate_records(records):
    """Validate record schema, IDs, labels, splits, and group isolation."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen_record_ids = set()
    group_splits = {}
    present_splits = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")

        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing fields: {sorted(missing)}")

        record_id = record["record_id"]
        group_id = record["group_id"]
        split = record["split"]
        label = record["label"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a non-empty string")

        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split for {record_id}: {split}")

        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"label for {record_id} must be 0 or 1")

        preprocess_image(record["pixels"])

        previous_split = group_splits.get(group_id)
        if previous_split is not None and previous_split != split:
            raise ValueError(
                f"group {group_id} appears in both {previous_split} and {split}"
            )

        group_splits[group_id] = split
        present_splits.add(split)

    if present_splits != VALID_SPLITS:
        missing_splits = VALID_SPLITS - present_splits
        raise ValueError(f"missing required splits: {sorted(missing_splits)}")

    return True


def load_fixture(path):
    """Load and validate a JSON fixture."""
    with Path(path).open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    if fixture.get("image_shape") != list(IMAGE_SHAPE):
        raise ValueError(f"fixture image_shape must be {list(IMAGE_SHAPE)}")

    if fixture.get("pixel_range") != [PIXEL_MIN, PIXEL_MAX]:
        raise ValueError("fixture pixel_range must be [0, 255]")

    validate_records(fixture.get("records"))
    return fixture


def transform_split(records, split):
    """Transform records from one split; this operation learns no statistics."""
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")

    validate_records(records)
    selected = [record for record in records if record["split"] == split]

    if not selected:
        raise ValueError(f"no records found for split: {split}")

    images = np.stack([preprocess_image(record["pixels"]) for record in selected])
    labels = np.asarray([record["label"] for record in selected], dtype=np.int64)
    return images, labels


if __name__ == "__main__":
    fixture_path = Path(__file__).with_name("fixture.json")
    data = load_fixture(fixture_path)

    for split_name in ("train", "validation", "test"):
        images, labels = transform_split(data["records"], split_name)
        print(
            f"{split_name}: images={images.shape}, "
            f"dtype={images.dtype}, labels={labels.tolist()}"
        )
