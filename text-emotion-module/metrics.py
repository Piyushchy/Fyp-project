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

from config import ID2LABEL


# ============================================================
# AGGREGATE METRICS
# ============================================================

def compute_metrics(labels, predictions, id2label=None):
    """Accuracy plus macro precision / recall / F1, and per-class scores.

    id2label selects the label space. It defaults to the six-class
    MTEB one so existing callers are unaffected; the GoEmotions task
    passes its own seven-class map. Scoring a model against the wrong
    label space would silently produce plausible-looking numbers, so
    the caller is made to be explicit about which one it means.
    """

    id2label = ID2LABEL if id2label is None else id2label

    num_labels = len(id2label)

    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(num_labels)),
            zero_division=0,
        )
    )

    # Every aggregate is averaged over the full label set, not just
    # the classes that happen to appear. Otherwise a model that never
    # predicts 'surprise' would be scored on the remaining classes
    # and look better than it is.
    all_labels = list(range(num_labels))

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
            id2label[index]: {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(num_labels)
        },
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def compute_confusion_matrix(labels, predictions, id2label=None):
    """Raw counts, rows = true label, columns = predicted label."""

    id2label = ID2LABEL if id2label is None else id2label

    return confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(id2label))),
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
