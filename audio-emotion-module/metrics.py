# ============================================================
# AUDIO EMOTION MODULE - METRICS
# ============================================================
#
# One scoring surface shared by training, benchmarking and the
# multimodal evaluation, so a number quoted in one place means
# the same thing in the others.
#
# What is reported and why
# ------------------------
#   accuracy        the headline, and on its own misleading here:
#                   the test split is left at its natural
#                   distribution, so a model that always answers
#                   neutral already scores well above chance.
#   macro_f1        the number this module is actually tuned on.
#                   Averaging F1 over classes without weighting by
#                   support means the 200-clip disgust class counts
#                   as much as the 2000-clip neutral one, which is
#                   the behaviour the fusion needs.
#   weighted_f1     the figure the MELD literature quotes, kept so
#                   the wild-domain result is comparable to it.
#   balanced_acc    mean per-class recall. Reported next to macro_f1
#                   because the two disagree exactly when the model
#                   is over-predicting a rare class - recall rises
#                   while precision falls.
#   per_class_f1    feeds fusion's reliability.json directly.
#   ece             expected calibration error. A modality that is
#                   wrong but uncertain costs a log-pool far less
#                   than one that is wrong and confident, so
#                   calibration is a fusion input, not a footnote.
# ============================================================

from __future__ import annotations

import numpy as np

from config import LABELS


# ============================================================
# CALIBRATION
# ============================================================

def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    """Gap between confidence and accuracy, averaged over bins.

    Predictions are bucketed by their top-class probability. Within a
    bucket, a perfectly calibrated model's accuracy equals its mean
    confidence; ECE is the support-weighted mean of |accuracy - mean
    confidence| across buckets.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)

    if probabilities.size == 0:
        return 0.0

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    # Right-closed edges so a confidence of exactly 1.0 lands in the
    # top bin rather than falling outside every one of them.
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())

        if count == 0:
            continue

        total += (count / len(labels)) * abs(
            correct[in_bin].mean() - confidence[in_bin].mean()
        )

    return float(total)


# ============================================================
# THE SCORE SHEET
# ============================================================

def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    """Everything the module reports about one set of predictions."""

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.size == 0:
        return {"support": 0}

    # Pin the class list so a split that happens to contain no disgust
    # still produces a seven-entry per-class table, with 0.0 in the
    # empty slot rather than a silently shorter array that would
    # misalign against the shared label space downstream.
    indices = list(range(len(LABELS)))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )

    result = {
        "support": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=indices, average="macro",
                     zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, labels=indices, average="weighted",
                     zero_division=0)
        ),
        "per_class_f1": {
            name: float(value) for name, value in zip(LABELS, f1)
        },
        "per_class_precision": {
            name: float(value) for name, value in zip(LABELS, precision)
        },
        "per_class_recall": {
            name: float(value) for name, value in zip(LABELS, recall)
        },
        "per_class_support": {
            name: int(value) for name, value in zip(LABELS, support)
        },
    }

    if probabilities is not None:
        result["ece"] = expected_calibration_error(probabilities, labels)
        result["mean_confidence"] = float(
            np.asarray(probabilities).max(axis=1).mean()
        )

    return result


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Rows are true classes, columns predicted, in shared order."""

    from sklearn.metrics import confusion_matrix

    return confusion_matrix(
        labels, predictions, labels=list(range(len(LABELS)))
    )


# ============================================================
# PRESENTATION
# ============================================================

def format_report(metrics: dict, title: str = "") -> str:
    """The per-class table, for logs and the console."""

    if not metrics.get("support"):
        return f"{title}\n  (empty split)"

    lines = []

    if title:
        lines.append(title)

    lines.append(
        f"  support {metrics['support']}   "
        f"acc {metrics['accuracy']:.4f}   "
        f"macro-F1 {metrics['macro_f1']:.4f}   "
        f"weighted-F1 {metrics['weighted_f1']:.4f}   "
        f"bal-acc {metrics['balanced_accuracy']:.4f}"
    )

    if "ece" in metrics:
        lines.append(
            f"  ECE {metrics['ece']:.4f}   "
            f"mean confidence {metrics['mean_confidence']:.4f}"
        )

    lines.append(f"  {'class':<10} {'prec':>7} {'rec':>7} {'f1':>7} {'n':>7}")

    for name in LABELS:
        lines.append(
            f"  {name:<10} "
            f"{metrics['per_class_precision'][name]:>7.4f} "
            f"{metrics['per_class_recall'][name]:>7.4f} "
            f"{metrics['per_class_f1'][name]:>7.4f} "
            f"{metrics['per_class_support'][name]:>7d}"
        )

    return "\n".join(lines)


def format_confusion(matrix: np.ndarray) -> str:
    """Confusion matrix with the shared labels down the side."""

    header = "  " + " " * 10 + "".join(f"{name[:5]:>7}" for name in LABELS)
    lines = [header]

    for name, row in zip(LABELS, matrix):
        lines.append(f"  {name:<10}" + "".join(f"{int(v):>7}" for v in row))

    return "\n".join(lines)
