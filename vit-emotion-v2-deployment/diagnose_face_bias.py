# ============================================================
# FACE BIAS DIAGNOSTIC
# ============================================================
#
# Why does the live session call so many faces 'disgust' or
# 'surprise' when they are plainly neutral or happy?
#
#   python diagnose_face_bias.py --data eval_data_unseen/test/webcam
#
# results_face_eval.*.json records per-class F1 and nothing else,
# which cannot answer that question. F1 collapses precision and
# recall into one number, so a class that is predicted five times
# too often and a class that is never predicted at all can carry
# the same score. What is needed is the confusion matrix and,
# more directly, the comparison between how often each label
# OCCURS and how often it is PREDICTED.
#
# This script reuses eval_face.py's pipeline so the numbers refer
# to the configuration server.py actually serves - aligned crop,
# multi-crop averaging, stored temperature, stored logit bias -
# rather than to a bare forward pass.
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_face as ef
from labels import SHARED_LABELS

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# SCORING
# ============================================================

def predictions_from(
    result: dict,
    temperature: float,
    bias: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(labels, predictions) with undetected faces dropped.

    A detector miss has no predicted class, so it cannot appear in a
    confusion matrix. eval_face counts it against accuracy, which is
    right for a headline number and wrong here: this script is asking
    which class the MODEL picks when it picks one.

    `_logits` holds the RAW network output - run_configuration applies
    temperature and bias afterwards, when it builds probabilities. So
    both have to be re-applied here. Temperature alone cannot change
    an argmax (it scales every logit by the same positive factor), but
    the per-class bias very much can, and reading the raw logits would
    silently report the uncalibrated model under the deployed label.
    """

    offset = (
        np.zeros(len(SHARED_LABELS), dtype=np.float32)
        if bias is None
        else np.asarray(bias, dtype=np.float32)
    )

    labels = []
    predictions = []

    for logits, label in zip(result["_logits"], result["_labels"]):
        if logits is None:
            continue

        adjusted = np.asarray(logits, dtype=np.float64) / temperature + offset

        labels.append(int(label))
        predictions.append(int(np.argmax(adjusted)))

    return np.array(labels), np.array(predictions)


def confusion_of(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    size = len(SHARED_LABELS)
    matrix = np.zeros((size, size), dtype=int)

    for true, predicted in zip(labels, predictions):
        matrix[true, predicted] += 1

    return matrix


def report(title: str, labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_of(labels, predictions)

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    correct = np.diag(matrix)

    precision = np.divide(
        correct, predicted_count,
        out=np.zeros(len(SHARED_LABELS)), where=predicted_count > 0,
    )
    recall = np.divide(
        correct, support,
        out=np.zeros(len(SHARED_LABELS)), where=support > 0,
    )

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    print("\nconfusion (rows true, columns predicted)")
    print("            " + "".join(f"{n[:5]:>7}" for n in SHARED_LABELS))

    for name, row in zip(SHARED_LABELS, matrix):
        print(f"  {name:<10}" + "".join(f"{v:>7}" for v in row))

    # ----------------------------------------------------
    # The number that answers the question
    # ----------------------------------------------------
    #
    # over-prediction ratio = times predicted / times it occurs.
    # 1.0 is a model whose output distribution matches reality.
    # Above 1.0 the class is being handed out too freely, and the
    # faces it steals have to come from somewhere.
    # ----------------------------------------------------

    print("\n  class        occurs  predicted   ratio   precision   recall")

    ratios = {}

    for index, name in enumerate(SHARED_LABELS):
        ratio = predicted_count[index] / max(1, support[index])
        ratios[name] = float(ratio)

        flag = ""

        if ratio >= 1.5:
            flag = "  <-- OVER"
        elif ratio <= 0.6:
            flag = "  <-- UNDER"

        print(
            f"  {name:<10} {support[index]:>7} {predicted_count[index]:>10} "
            f"{ratio:>7.2f} {precision[index]:>11.3f} {recall[index]:>8.3f}"
            f"{flag}"
        )

    accuracy = float((labels == predictions).mean())

    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(len(SHARED_LABELS)), where=(precision + recall) > 0,
    )

    print(f"\n  accuracy {accuracy:.4f}   macro-F1 {f1.mean():.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "confusion": matrix.tolist(),
        "ratios": ratios,
        "precision": {n: float(v) for n, v in zip(SHARED_LABELS, precision)},
        "recall": {n: float(v) for n, v in zip(SHARED_LABELS, recall)},
        "f1": {n: float(v) for n, v in zip(SHARED_LABELS, f1)},
    }


# ============================================================
# WHERE THE WRONG CALLS GO
# ============================================================

def leakage(matrix: np.ndarray) -> None:
    """For each true class, the wrong label it most often receives."""

    print("\n  true class -> most common WRONG prediction")

    for index, name in enumerate(SHARED_LABELS):
        row = matrix[index].astype(float).copy()
        total = row.sum()

        if total == 0:
            continue

        row[index] = -1  # ignore the diagonal

        worst = int(np.argmax(row))

        if row[worst] <= 0:
            continue

        print(
            f"    {name:<10} -> {SHARED_LABELS[worst]:<10} "
            f"{int(row[worst]):>4} / {int(total)}  "
            f"({row[worst] / total:.1%} of all {name} faces)"
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args()

    samples = ef.load_index(arguments.data, arguments.limit_per_class)

    print(f"{len(samples)} images from {arguments.data}")

    classifier, config = ef.build_classifier(prefer_gpu=not arguments.cpu)
    detector = ef.build_detector()

    temperature = float(config.get("temperature", 1.0))
    stored_bias = config.get("logit_bias")

    print(f"model      {ef.ONNX_PATH.name}")
    print(f"T          {temperature:.4f}")
    print(f"logit_bias {stored_bias}")

    findings = {}

    # ----------------------------------------------------
    # 1. What the server actually serves
    # ----------------------------------------------------

    deployed = ef.run_configuration(
        ef.Configuration(
            "deployed", ef.CROP_MODE_DEPLOYED, ef.USE_TTA_DEPLOYED,
            temperature,
            "as served",
            logit_bias=(
                None if stored_bias is None
                else np.asarray(stored_bias, dtype=np.float32)
            ),
            multicrop=ef.USE_MULTICROP_DEPLOYED,
        ),
        samples, detector, classifier,
    )

    labels, predictions = predictions_from(
        deployed,
        temperature,
        None if stored_bias is None else np.asarray(stored_bias),
    )
    findings["deployed"] = report(
        "DEPLOYED PIPELINE (multi-crop + T + stored logit bias)",
        labels, predictions,
    )

    leakage(np.array(findings["deployed"]["confusion"]))

    # ----------------------------------------------------
    # 2. The same model with the bias removed
    # ----------------------------------------------------
    #
    # Isolates how much of the skew the stored bias is causing
    # versus how much the model brings on its own.
    # ----------------------------------------------------

    nobias = ef.run_configuration(
        ef.Configuration(
            "no bias", ef.CROP_MODE_DEPLOYED, ef.USE_TTA_DEPLOYED,
            temperature,
            "bias zeroed",
            logit_bias=None,
            multicrop=ef.USE_MULTICROP_DEPLOYED,
        ),
        samples, detector, classifier,
    )

    labels_nb, predictions_nb = predictions_from(nobias, temperature, None)
    findings["no_bias"] = report(
        "SAME MODEL, STORED LOGIT BIAS REMOVED", labels_nb, predictions_nb,
    )

    output = BASE_DIR / "results_face_bias.json"

    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data": str(arguments.data),
                "model": ef.ONNX_PATH.name,
                "temperature": temperature,
                "logit_bias": stored_bias,
                "findings": findings,
            },
            handle, indent=2,
        )

    print(f"\nwrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
