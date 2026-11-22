
import json
from pathlib import Path

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.json"
SCHEMA_VERSION = "1.0"


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Records must be a non-empty list")

    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        if set(record) != {"record_id", "text", "label"}:
            raise ValueError("Invalid record schema")
        if not isinstance(record["record_id"], str) or not record["record_id"]:
            raise ValueError("Invalid record_id")
        if record["record_id"] in seen:
            raise ValueError("Duplicate record_id")
        seen.add(record["record_id"])
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError("Invalid text")
        if type(record["label"]) is not int or record["label"] not in (0, 1):
            raise ValueError("Invalid label")


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError("Model must be a JSON object")
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported schema version")
    if model.get("pipeline_type") != "deterministic_bigram_demo":
        raise ValueError("Unsupported pipeline type")
    if model.get("parameters") != {
        "lowercase": True, "unknown_token": "<UNK>"
    }:
        raise ValueError("Invalid pipeline parameters")

    mapping = model.get("next_words")
    if not isinstance(mapping, dict):
        raise ValueError("next_words must be an object")
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Invalid next-word mapping")


def build_pipeline(records):
    validate_records(records)
    next_words = {}
    for record in records:
        words = record["text"].lower().split()
        for first, second in zip(words, words[1:]):
            next_words.setdefault(first, second)

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_type": "deterministic_bigram_demo",
        "parameters": {"lowercase": True, "unknown_token": "<UNK>"},
        "next_words": next_words
    }


def save_pipeline(model, path=MODEL_PATH):
    validate_model(model)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2, sort_keys=True)


def load_pipeline(path=MODEL_PATH):
    with open(path, encoding="utf-8") as file:
        model = json.load(file)
    validate_model(model)
    return model


def predict_next(model, text):
    validate_model(model)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    words = text.lower().split()
    return model["next_words"].get(words[-1], "<UNK>")


def run_demo():
    with open(BASE / "fixture.json", encoding="utf-8") as file:
        fixture = json.load(file)

    records = fixture["records"]
    validate_records(records)

    train_ids = set(fixture["splits"]["train"])
    train_records = [
        record for record in records
        if record["record_id"] in train_ids
    ]

    model = build_pipeline(train_records)
    before = predict_next(model, "red")
    save_pipeline(model)
    reloaded = load_pipeline()
    after = predict_next(reloaded, "red")

    return {
        "record_count": len(records),
        "training_record_count": len(train_records),
        "prediction_before_save": before,
        "prediction_after_reload": after,
        "predictions_match": before == after
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))

