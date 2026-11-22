
import json
from pathlib import Path


def classification_metrics(truth, prediction):
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have equal non-zero lengths")

    if any(value not in (0, 1) for value in truth + prediction):
        raise ValueError("classification values must be binary (0 or 1)")

    tp = sum(t == 1 and p == 1 for t, p in zip(truth, prediction))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, prediction))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, prediction))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "averaging": "binary",
        "zero_denominator": 0.0,
    }


def mean_absolute_error(truth, prediction):
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have equal non-zero lengths")

    return sum(abs(t - p) for t, p in zip(truth, prediction)) / len(truth)


def load_fixture():
    path = Path(__file__).with_name("fixture.json")
    with path.open(encoding="utf-8") as file:
        fixture = json.load(file)

    if fixture["sample_rate"] != 8000:
        raise ValueError("sample_rate must be 8000")

    records = fixture["records"]
    if len(records) != 6:
        raise ValueError("fixture must contain six records")

    ids = [record["record_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record IDs must be unique")

    for record in records:
        if record["sample_rate"] != 8000 or len(record["samples"]) != 8:
            raise ValueError("each waveform must have eight samples at 8000 Hz")

    return fixture


if __name__ == "__main__":
    fixture = load_fixture()
    classification = classification_metrics(
        [1, 1, 0, 0], [1, 0, 1, 0]
    )
    mae = mean_absolute_error([1, 2, 3], [1, 3, 2])

    print(json.dumps({
        "records": len(fixture["records"]),
        "classification": classification,
        "regression_mae": mae,
        "note": "Synthetic pipeline verification only; not real ESC-50 performance."
    }, indent=2))
