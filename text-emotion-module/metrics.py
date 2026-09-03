# ============================================================
# TEXT EMOTION MODULE - METRICS
# ============================================================
#
# One metric definition shared by training and benchmarking so
# the numbers in both places are computed identically.
#
# Macro averaging is the headline metric. MTEB emotion is very
# imbalanced (joy 5345 rows, surprise 568), so micro/weighted
# scores mostly measure performance on the two largest classes.
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from config import ID2LABEL, NUM_LABELS


# ============================================================
# AGGREGATE METRICS
# ============================================================

def compute_metrics(labels, predictions):
    """Accuracy plus macro precision / recall / F1, and per-class scores."""

    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(NUM_LABELS)),
            zero_division=0,
        )
    )

    # Every aggregate is averaged over the full six-class label
    # set, not just the classes that happen to appear. Otherwise
    # a model that never predicts 'surprise' would be scored on
    # five classes and look better than it is.
    all_labels = list(range(NUM_LABELS))

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(
                labels, predictions,
                labels=all_labels, average="macro", zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels, predictions,
                labels=all_labels, average="macro", zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels, predictions,
                labels=all_labels, average="macro", zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels, predictions,
                labels=all_labels, average="weighted", zero_division=0,
            )
        ),
        "per_class": {
            ID2LABEL[index]: {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(NUM_LABELS)
        },
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def compute_confusion_matrix(labels, predictions):
    """Raw counts, rows = true label, columns = predicted label."""

    return confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_LABELS)),
    )


# ============================================================
# TEXT REPORT
# ============================================================

def format_report(metrics, title=""):
    """Render one metrics dict as an aligned text table."""

    lines = []

    if title:
        lines.append(title)

    lines.append(f"Accuracy         {metrics['accuracy']*100:6.2f}%")
    lines.append(f"Macro Precision  {metrics['macro_precision']*100:6.2f}%")
    lines.append(f"Macro Recall     {metrics['macro_recall']*100:6.2f}%")
    lines.append(f"Macro F1         {metrics['macro_f1']*100:6.2f}%")
    lines.append("")
    lines.append(f"{'Emotion':<10} {'Precision':>10} {'Recall':>9} "
                 f"{'F1':>8} {'Support':>8}")

    for name, scores in metrics["per_class"].items():
        lines.append(
            f"{name:<10} {scores['precision']*100:9.2f}% "
            f"{scores['recall']*100:8.2f}% {scores['f1']*100:7.2f}% "
            f"{scores['support']:8d}"
        )

    return "\n".join(lines)
