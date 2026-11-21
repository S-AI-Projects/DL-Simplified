"""Validation utilities for the R42-T011 text pipeline."""

REQUIRED_FIELDS = {"record_id", "text", "label"}


def validate_record(record):
    errors = []

    if not isinstance(record, dict):
        return ["unknown | record | must be a dictionary"]

    record_id = record.get("record_id", "unknown")

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(
                f"{record_id} | {field} | missing required field"
            )

    if "record_id" in record:
        value = record["record_id"]
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{record_id} | record_id | must be a non-empty string"
            )

    if "text" in record:
        value = record["text"]
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{record_id} | text | must be non-empty text"
            )

    if "label" in record:
        value = record["label"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(
                f"{record_id} | label | must be numeric"
            )

    return errors


def validate_records(records):
    errors = []
    seen_ids = set()

    for record in records:
        errors.extend(validate_record(record))

        if isinstance(record, dict):
            record_id = record.get("record_id")

            if isinstance(record_id, str):
                if record_id in seen_ids:
                    errors.append(
                        f"{record_id} | record_id | duplicate record ID"
                    )
                seen_ids.add(record_id)

    return errors