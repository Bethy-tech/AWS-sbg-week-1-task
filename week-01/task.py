"""Student score data-cleaning workflow.

This script demonstrates a simple data-processing pipeline for a small CSV file.
It loads a raw dataset, standardizes text, converts values to useful Python data
types, cleans invalid values, removes duplicates, and calculates summary
statistics with NumPy.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

VALID_AGE_MIN = 0
VALID_AGE_MAX = 100
VALID_SCORE_MIN = 0
VALID_SCORE_MAX = 100


def load_csv_dataset(file_path: str | Path) -> List[Dict[str, Any]]:
    """Load the CSV file and return a list of dictionaries.

    Missing files are handled gracefully by returning an empty list.
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        print(f"Warning: Dataset not found at {csv_path}")
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def standardize_text(value: Any) -> str:
    """Trim spaces and fix capitalization for text fields."""
    if value is None:
        return ""

    text = str(value).strip()
    text = " ".join(text.split())
    return text.title()


def convert_to_number(value: Any) -> float:
    """Convert a value to a float if it is a valid number.

    If the value is missing or not numeric, a NaN is returned so the value can be
    handled later in a controlled way.
    """
    if value is None:
        return float("nan")

    text = str(value).strip()
    if text == "":
        return float("nan")

    cleaned_text = text.replace(",", "")

    try:
        return float(cleaned_text)
    except ValueError:
        return float("nan")


def standardize_text_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean text values such as names and other labels."""
    cleaned = []

    for row in records:
        cleaned_row = dict(row)
        cleaned_row["name"] = standardize_text(cleaned_row.get("name"))
        cleaned.append(cleaned_row)

    return cleaned


def convert_data_types(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert text-based values to numeric types when appropriate."""
    converted = []

    for row in records:
        cleaned_row = dict(row)
        cleaned_row["age"] = convert_to_number(cleaned_row.get("age"))
        cleaned_row["score"] = convert_to_number(cleaned_row.get("score"))
        converted.append(cleaned_row)

    return converted


def handle_missing_values(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill missing values in a safe and explainable way.

    For this dataset, we use the median for missing numeric values. This keeps the
    data centered and avoids the effect of extreme values.
    """
    if not records:
        return []

    ages = [row["age"] for row in records if not np.isnan(row["age"])]
    scores = [row["score"] for row in records if not np.isnan(row["score"])]

    age_median = float(np.median(ages)) if ages else 0.0
    score_median = float(np.median(scores)) if scores else 0.0

    fixed = []
    for row in records:
        new_row = dict(row)

        if not new_row.get("name"):
            new_row["name"] = "Unknown Student"

        if np.isnan(new_row["age"]):
            new_row["age"] = age_median

        if np.isnan(new_row["score"]):
            new_row["score"] = score_median

        fixed.append(new_row)

    return fixed


def remove_duplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact duplicate records based on normalized values."""
    seen = set()
    unique_records = []

    for row in records:
        key = (
            standardize_text(row.get("name")).lower(),
            float(row["age"]),
            float(row["score"]),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_records.append(row)

    return unique_records


def detect_and_handle_invalid_values(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix age and score values that are unrealistic or non-numeric.

    We use reasonable data-specific rules:
    - age should usually be between 0 and 100 for a student dataset
    - score should be between 0 and 100

    Invalid values are replaced by the median value from that column.
    """
    if not records:
        return []

    ages = [row["age"] for row in records]
    scores = [row["score"] for row in records]

    age_median = float(np.median(ages)) if ages else 0.0
    score_median = float(np.median(scores)) if scores else 0.0

    corrected = []
    for row in records:
        new_row = dict(row)

        if not np.isfinite(new_row["age"]) or new_row["age"] < VALID_AGE_MIN or new_row["age"] > VALID_AGE_MAX:
            new_row["age"] = age_median

        if not np.isfinite(new_row["score"]) or new_row["score"] < VALID_SCORE_MIN or new_row["score"] > VALID_SCORE_MAX:
            new_row["score"] = score_median

        corrected.append(new_row)

    return corrected


def calculate_statistics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute useful summary statistics from the cleaned score column."""
    if not records:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    scores = np.array([float(row["score"]) for row in records], dtype=float)

    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std_dev": float(np.std(scores)),
        "p90": float(np.percentile(scores, 90)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def clean_student_data(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the full cleaning workflow in a logical order."""
    standardized_records = standardize_text_fields(raw_records)
    typed_records = convert_data_types(standardized_records)
    filled_records = handle_missing_values(typed_records)
    deduplicated_records = remove_duplicate_records(filled_records)
    validated_records = detect_and_handle_invalid_values(deduplicated_records)
    return validated_records


def print_statistics(stats: Dict[str, float]) -> None:
    """Print the statistics in a beginner-friendly format."""
    print("\nSummary statistics for student scores:")
    print(f"Mean score: {stats['mean']:.2f}")
    print(f"Median score: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std_dev']:.2f}")
    print(f"90th percentile: {stats['p90']:.2f}")
    print(f"Minimum score: {stats['min']:.2f}")
    print(f"Maximum score: {stats['max']:.2f}")


def main() -> None:
    """Run the complete CSV processing workflow."""
    dataset_path = Path(__file__).resolve().parent / "students.csv"

    raw_records = load_csv_dataset(dataset_path)
    print(f"Raw records loaded: {len(raw_records)}")

    cleaned_records = clean_student_data(raw_records)
    print(f"Cleaned records: {len(cleaned_records)}")

    if cleaned_records:
        for index, record in enumerate(cleaned_records, start=1):
            print(f"{index}. {record['name']} | age={record['age']:.0f} | score={record['score']:.2f}")

    stats = calculate_statistics(cleaned_records)
    print_statistics(stats)


if __name__ == "__main__":
    main()
