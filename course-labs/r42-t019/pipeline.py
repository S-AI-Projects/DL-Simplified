import json
import random
from pathlib import Path

BASE = Path(__file__).parent
SEED = 42


def load_fixture(path=BASE / "fixture.json"):
    with open(path, encoding="utf-8") as file:
        fixture = json.load(file)

    if fixture.get("image_shape") != [4, 4]:
        raise ValueError("Expected 4x4 images")

    records = fixture.get("records", [])
    if len(records) != 6:
        raise ValueError("Expected six records")

    ids = [r["record_id"] for r in records]
    groups = [r["group_id"] for r in records]
    if len(set(ids)) != 6 or len(set(groups)) != 6:
        raise ValueError("Record and group IDs must be unique")

    for record in records:
        if set(record) != {"record_id", "group_id", "pixels", "label"}:
            raise ValueError("Invalid record fields")
        if record["label"] not in (0, 1):
            raise ValueError("Invalid label")

        pixels = record["pixels"]
        if len(pixels) != 4 or any(len(row) != 4 for row in pixels):
            raise ValueError("Expected 4x4 pixel arrays")

        for row in pixels:
            for pixel in row:
                if type(pixel) is not int or not 0 <= pixel <= 255:
                    raise ValueError("Invalid pixel value")

    return records


def predict(record):
    return int(any(pixel > 0 for row in record["pixels"] for pixel in row))


def run_pipeline(fixture_path=BASE / "fixture.json"):
    records = load_fixture(fixture_path)

    baseline = {
        record["record_id"]: predict(record)
        for record in records
    }

    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)

    perturbed = {
        record["record_id"]: predict(record)
        for record in shuffled
    }

    return {
        "seed": SEED,
        "record_count": len(records),
        "order_changed": [
            r["record_id"] for r in records
        ] != [
            r["record_id"] for r in shuffled
        ],
        "baseline_predictions": baseline,
        "shuffled_predictions": perturbed,
        "stable_after_alignment": baseline == perturbed
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
