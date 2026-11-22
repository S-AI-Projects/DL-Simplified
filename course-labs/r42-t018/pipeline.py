import json
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
        raise ValueError("Expected exactly six records")

    ids = [record["record_id"] for record in records]
    groups = [record["group_id"] for record in records]

    if len(set(ids)) != 6 or len(set(groups)) != 6:
        raise ValueError("Record and group IDs must be unique")

    for record in records:
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


def analyze(records):
    rows = []

    for record in records:
        expected = record["label"]
        predicted = predict(record)

        # Deliberately inject one incorrect prediction for testing.
        if record["record_id"] == "diagonal-003":
            predicted = 0

        if predicted == expected:
            continue

        rows.append({
            "record_id": record["record_id"],
            "expected": expected,
            "predicted": predicted,
            "reason_category": "injected_prediction_error"
        })

    return {
        "seed": SEED,
        "total_records": len(records),
        "error_count": len(rows),
        "correct_count": len(records) - len(rows),
        "errors": rows
    }


def run_pipeline(fixture_path=BASE / "fixture.json"):
    return analyze(load_fixture(fixture_path))


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
