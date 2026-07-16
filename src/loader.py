"""
Dataset and taxonomy loading for classification experiments.

All paths come from the caller (typically config) — this module only
reads and validates, never hardcodes locations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("ticket", "label")


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a ticket CSV and validate required columns.

    Args:
        path: Path to a CSV with at least `ticket` and `label` columns.

    Returns:
        DataFrame with the ticket records.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Set DATASET_PATH in .env to a valid CSV under datasets/raw/."
        )

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing)}.\n"
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}.\n"
            f"Found columns: {', '.join(df.columns.astype(str))}."
        )

    return df


def load_taxonomy(path: Path) -> dict[str, str]:
    """
    Load the category taxonomy from a JSON object file.

    Expected shape: { "CategoryName": "description", ... }
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy not found: {path}\n"
            f"Set TAXONOMY_PATH in .env to a valid JSON file under datasets/raw/."
        )

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"Taxonomy at {path} must be a non-empty JSON object "
            f"mapping category names to descriptions."
        )

    return {str(key): str(value) for key, value in data.items()}


def validate_labels(df: pd.DataFrame, taxonomy: dict[str, str]) -> None:
    """
    Ensure every label in the dataset exists as a taxonomy key.

    Raises:
        ValueError: Listing unknown labels so the dataset can be fixed quickly.
    """
    labels = set(df["label"].astype(str).unique())
    known = set(taxonomy.keys())
    unknown = sorted(labels - known)

    if unknown:
        raise ValueError(
            "Dataset contains label(s) not present in the taxonomy:\n"
            + "\n".join(f"  - {label}" for label in unknown)
            + "\n\nValid taxonomy labels:\n"
            + "\n".join(f"  - {label}" for label in sorted(known))
        )
