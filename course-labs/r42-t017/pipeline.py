import json
from pathlib import Path


BASE = Path(__file__).parent

CONFIG_KEYS = {
    "schema_version", "seed", "feature_order", "split_ids", "algorithm"
}
ALGORITHM_KEYS = {"name", "threshold", "max_samples", "max_steps"}
SPLITS = {"train", "validation", "test"}


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_config(path=BASE / "config.json"):
    config = load_json(path)

    if set(config) != CONFIG_KEYS:
        raise ValueError("Unknown or missing configuration fields")

    if config["schema_version"] != "1.0":
        raise ValueError("Unsupported schema version")

    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Invalid seed")

    if config["feature_order"] != ["row_major_flattened_pixels"]:
        raise ValueError("Unsupported feature ordering")

    splits = config["split_ids"]
    if set(splits) != SPLITS:
        raise ValueError("Invalid split definitions")

    all_ids = []
    for ids in splits.values():
        if not isinstance(ids, list) or not ids:
            raise ValueError("Every split must contain records")
        all_ids.extend(ids)

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Overlapping or duplicate split IDs")

    algorithm = config["algorithm"]
    if set(algorithm) != ALGORITHM_KEYS:
        raise ValueError("Unknown or missing algorithm settings")

    if algorithm["name"] != "nonzero_pixel_baseline":
        raise ValueError("Unsupported algorithm")

    if type(algorithm["threshold"]) is not int or algorithm["threshold"] != 0:
        raise ValueError("Invalid threshold")

    if type(algorithm["max_samples"]) is not int or not 1 <= algorithm["max_samples"] <= 32:
        raise ValueError("Invalid sample limit")

    if type(algorithm["max_steps"]) is not int or not 1 <= algorithm["max_steps"] <= 20:
        raise ValueError("Invalid step limit")

    return config


def load_fixture(path=BASE / "fixture.json"):
    fixture = load_json(path)
    if fixture["image_shape"] != [4, 4]:
        raise ValueError("Incompatible image shape")

    if fixture["pixel_range"] != [0, 255]:
        raise ValueError("Invalid pixel range")

    records = fixture["records"]
    if len(records) != 6:
        raise ValueError("Expected exactly six records")

    record_ids = [r["record_id"] for r in records]
    group_ids = [r["group_id"] for r in records]

    if len(set(record_ids)) != 6 or len(set(group_ids)) != 6:
        raise ValueError("Record and group IDs must be unique")

    for record in records:
        if len(record["pixels"]) != 4 or any(len(row) != 4 for row in record["pixels"]):
            raise ValueError("Incompatible input shape")

        if record["label"] not in (0, 1):
            raise ValueError("Invalid label")

        for row in record["pixels"]:
            for pixel in row:
                if type(pixel) is not int or not 0 <= pixel <= 255:
                    raise ValueError("Invalid pixel value")

    return records


def run_pipeline(config_path=BASE / "config.json",
                 fixture_path=BASE / "fixture.json"):
    config = load_config(config_path)
    records = load_fixture(fixture_path)

    by_id = {record["record_id"]: record for record in records}
    configured_ids = [
        record_id
        for split in config["split_ids"].values()
        for record_id in split
    ]

    if set(configured_ids) != set(by_id):
        raise ValueError("Configuration IDs do not match fixture")

    if len(configured_ids) > config["algorithm"]["max_samples"]:
        raise ValueError("Sample limit exceeded")

    results = {}

    for split, ids in config["split_ids"].items():
        predictions = []
        actual = []

        for record_id in ids:
            record = by_id[record_id]
            features = [
                pixel
                for row in record["pixels"]
                for pixel in row
            ]

            prediction = int(
                any(pixel > config["algorithm"]["threshold"]
                    for pixel in features)
            )

            predictions.append(prediction)
            actual.append(record["label"])

        results[split] = {
            "record_ids": ids,
            "predictions": predictions,
            "actual": actual
        }

    return {
        "seed": config["seed"],
        "schema_version": config["schema_version"],
        "algorithm": config["algorithm"]["name"],
        "results": results
    }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
