
import json
import random
from collections import defaultdict
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records):
    """Validate the image fixture before creating splits."""
    if not isinstance(records, list) or not records:
        raise ValueError("Records must be a non-empty list.")

    seen_ids = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a dictionary.")

        required = {"record_id", "group_id", "pixels", "label"}
        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing)}"
            )

        record_id = record["record_id"]
        group_id = record["group_id"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Record {index} has an invalid record_id.")

        if record_id in seen_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")

        seen_ids.add(record_id)

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{record_id} has an invalid group_id.")

        label = record["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{record_id} has an invalid label.")

        pixels = record["pixels"]

        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError(f"{record_id} must have 4 pixel rows.")

        for row in pixels:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError(f"{record_id} must have 4 pixels per row.")

            for pixel in row:
                if type(pixel) is not int or not 0 <= pixel <= 255:
                    raise ValueError(f"{record_id} has an invalid pixel value.")

    return records


def create_split_manifest(records, seed=42):
    """Create deterministic splits without separating groups."""
    validate_records(records)

    groups = defaultdict(list)

    for record in records:
        groups[record["group_id"]].append(record["record_id"])

    group_ids = sorted(groups)

    if len(group_ids) < 3:
        raise ValueError(
            "At least 3 distinct groups are required for three splits."
        )

    random.Random(seed).shuffle(group_ids)

    total_groups = len(group_ids)
    train_count = max(1, int(total_groups * 0.6))
    validation_count = max(1, int(total_groups * 0.2))

    # Keep at least one group for the test split.
    if train_count + validation_count >= total_groups:
        train_count = total_groups - 2
        validation_count = 1

    split_groups = {
        "train": group_ids[:train_count],
        "validation": group_ids[
            train_count:train_count + validation_count
        ],
        "test": group_ids[train_count + validation_count:],
    }

    splits = {}

    for split_name in SPLIT_NAMES:
        assigned_groups = sorted(split_groups[split_name])
        record_ids = sorted(
            record_id
            for group_id in assigned_groups
            for record_id in groups[group_id]
        )

        splits[split_name] = {
            "group_ids": assigned_groups,
            "record_ids": record_ids,
        }

    return {
        "seed": seed,
        "split_unit": "group_id",
        "splits": splits,
    }


def load_fixture(path):
    """Load and validate the JSON fixture."""
    with Path(path).open("r", encoding="utf-8") as file:
        fixture = json.load(file)

    if not isinstance(fixture, dict) or "records" not in fixture:
        raise ValueError("Fixture must contain a records field.")

    validate_records(fixture["records"])
    return fixture["records"]


if __name__ == "__main__":
    fixture_path = Path(__file__).with_name("fixture.json")
    records = load_fixture(fixture_path)
    manifest = create_split_manifest(records)

    print(json.dumps(manifest, indent=2))

