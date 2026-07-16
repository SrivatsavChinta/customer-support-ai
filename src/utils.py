"""
Shared helpers used across experiments.

Keep these small and dependency-light so every experiment can reuse them
without dragging in heavy frameworks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def get_env(name: str) -> str:
    """
    Read a required environment variable.

    Raises ValueError if the variable is missing or empty so misconfiguration
    fails fast instead of silently using wrong defaults.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(
            f"Environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value.strip()


def get_env_float(name: str) -> float:
    """Read a required environment variable and parse it as float."""
    return float(get_env(name))


def get_env_int(name: str) -> int:
    """Read a required environment variable and parse it as int."""
    return int(get_env(name))


def load_prompt(path: Path) -> str:
    """
    Load a prompt template from disk.

    Prompts live as plain text files so they can be iterated on without
    touching Python code.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure a simple console logger for experiment runs.

    Using a named logger (not root) keeps third-party library noise out of
    our experiment output.
    """
    logger = logging.getLogger("ai_product_lab")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def next_run_path(predictions_dir: Path) -> Path:
    """
    Return the next non-colliding run_XXX.csv path under predictions_dir.

    Creates the directory if needed so callers never overwrite prior runs.
    """
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_numbers: list[int] = []
    for path in predictions_dir.glob("run_*.csv"):
        suffix = path.stem.removeprefix("run_")
        if suffix.isdigit():
            run_numbers.append(int(suffix))

    next_number = max(run_numbers, default=0) + 1
    return predictions_dir / f"run_{next_number:03d}.csv"
