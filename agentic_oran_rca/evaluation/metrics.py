from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: list[str]
    confusion: np.ndarray
    report_text: str


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> MetricResult:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return MetricResult(
        accuracy=acc,
        precision_macro=float(p),
        recall_macro=float(r),
        f1_macro=float(f1),
        labels=labels,
        confusion=cm,
        report_text=rep,
    )

