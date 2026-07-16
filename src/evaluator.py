"""
Evaluation orchestration for classification experiments.

Compares model predictions against ground-truth labels and produces
summary scores. Logic will be filled in once a labeled dataset exists.
"""

from __future__ import annotations

from typing import Any


def evaluate(
    predictions: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """
    Run the full evaluation suite on a batch of predictions.

    TODO: Compute accuracy.
    TODO: Compute precision (macro / micro / per-class).
    TODO: Compute recall (macro / micro / per-class).
    TODO: Compute F1 score.
    TODO: Build a confusion matrix.
    TODO: Return a structured metrics dict suitable for saving under outputs/.
    """
    raise NotImplementedError(
        "Evaluation is not implemented yet. "
        "Add a labeled dataset and wire metrics in src/metrics.py first."
    )
