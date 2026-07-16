"""
Metric helpers for classification evaluation.

Pure functions that turn prediction/label lists into numbers.
Kept separate from evaluator.py so individual metrics can be unit-tested
and reused across future experiments.
"""

from __future__ import annotations

from typing import Any


def accuracy(predictions: list[str], labels: list[str]) -> float:
    """
    Fraction of predictions that match the ground-truth labels.

    TODO: Implement exact-match accuracy.
    """
    raise NotImplementedError("TODO: implement accuracy")


def precision(predictions: list[str], labels: list[str]) -> float:
    """
    Precision across classes (define macro vs micro when implementing).

    TODO: Implement precision.
    """
    raise NotImplementedError("TODO: implement precision")


def recall(predictions: list[str], labels: list[str]) -> float:
    """
    Recall across classes (define macro vs micro when implementing).

    TODO: Implement recall.
    """
    raise NotImplementedError("TODO: implement recall")


def f1_score(predictions: list[str], labels: list[str]) -> float:
    """
    Harmonic mean of precision and recall.

    TODO: Implement F1.
    """
    raise NotImplementedError("TODO: implement F1")


def confusion_matrix(
    predictions: list[str],
    labels: list[str],
) -> Any:
    """
    Confusion matrix over predicted vs true categories.

    TODO: Implement confusion matrix (return type TBD — nested dict or matrix).
    """
    raise NotImplementedError("TODO: implement confusion matrix")
