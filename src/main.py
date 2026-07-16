"""
Entry point for the baseline ticket-classification experiment.

Loads config + data, runs every ticket through the local Ollama classifier,
and writes a new predictions CSV under outputs/predictions/ (never overwrites).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import PROJECT_ROOT, config
from src.classifier import classify
from src.loader import load_dataset, load_taxonomy, validate_labels
from src.utils import next_run_path, setup_logging

logger = setup_logging()


def check_ollama() -> bool:
    """
    Return True if the local Ollama server responds.

    Hitting /api/tags is a lightweight health check that also confirms
    the server can list models — useful when debugging missing pulls.
    """
    url = f"{config.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Ollama is not reachable at %s (%s)", config.ollama_base_url, exc)
        return False


def check_prompt() -> bool:
    """Return True if the configured prompt file exists on disk."""
    if config.prompt_path.exists():
        return True
    logger.error("Prompt file not found: %s", config.prompt_path)
    return False


def print_banner(num_tickets: int) -> None:
    """Print the experiment startup banner."""
    print()
    print("======================================")
    print("Customer Support AI")
    print("======================================")
    print()
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    print(f"Dataset: {config.dataset_path}")
    print()
    print(f"Number of tickets: {num_tickets}")
    print()
    print("Processing...")
    print()


def print_summary(
    output_path: Path,
    total: int,
    successful: int,
    failed: int,
) -> None:
    """Print the experiment completion summary."""
    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path

    print()
    print("======================================")
    print()
    print("Experiment Complete")
    print()
    print("Predictions saved to:")
    print()
    print(str(display_path))
    print()
    print(f"Total Tickets: {total}")
    print()
    print(f"Successful: {successful}")
    print()
    print(f"Failed: {failed}")
    print()
    print("======================================")
    print()


def run_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Classify every ticket; never abort the batch on a single failure.

    Returns:
        (results_df, successful_count, failed_count)
    """
    rows: list[dict[str, str]] = []
    successful = 0
    failed = 0

    print("Processing tickets...")
    for _, record in tqdm(df.iterrows(), total=len(df), unit="ticket"):
        ticket = str(record["ticket"])
        actual_label = str(record["label"])

        try:
            result = classify(ticket)
            rows.append(
                {
                    "ticket": ticket,
                    "actual_label": actual_label,
                    "predicted_label": str(result.get("category", "")),
                    "reason": str(result.get("reason", "")),
                }
            )
            successful += 1
        except Exception as exc:
            # Keep going so one bad response does not waste the rest of the run
            logger.error("Failed to classify ticket %r: %s", ticket[:80], exc)
            rows.append(
                {
                    "ticket": ticket,
                    "actual_label": actual_label,
                    "predicted_label": "",
                    "reason": f"ERROR: {exc}",
                }
            )
            failed += 1

    results = pd.DataFrame(
        rows,
        columns=["ticket", "actual_label", "predicted_label", "reason"],
    )
    return results, successful, failed


def save_predictions(results: pd.DataFrame) -> Path:
    """Write results to the next run_XXX.csv under the predictions directory."""
    output_path = next_run_path(config.predictions_dir)
    results.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    """Run the baseline inference experiment end-to-end."""
    if not check_ollama():
        print()
        print("Ollama is unavailable. Start it locally and try again:")
        print("  ollama serve")
        print(f"  ollama pull {config.model_name}")
        print()
        return 1

    if not check_prompt():
        return 1

    if not config.dataset_path.exists():
        print()
        print("No dataset found.")
        print("Project setup complete.")
        print("Next step: create the first dataset.")
        print()
        return 0

    try:
        df = load_dataset(config.dataset_path)
        taxonomy = load_taxonomy(config.taxonomy_path)
        validate_labels(df, taxonomy)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1

    print_banner(len(df))
    results, successful, failed = run_batch(df)
    output_path = save_predictions(results)
    print_summary(output_path, total=len(df), successful=successful, failed=failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
