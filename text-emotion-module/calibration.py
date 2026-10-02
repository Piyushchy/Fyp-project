# ============================================================
# TEXT EMOTION MODULE - CALIBRATION
# ============================================================
#
# Temperature scaling, and the metric used to show it worked.
#
# Why this exists
# ---------------
# Both tasks train with an inverse-frequency class-weighted loss,
# because without it the rare classes are ignored and macro-F1
# collapses. The side effect is that the softmax output stops
# being a calibrated probability: the weighting pushes the
# decision boundary toward rare classes, so the model reports
# more confidence than its accuracy justifies.
#
# For a single-modality demo that would not matter - argmax is
# unaffected by a monotone rescaling, so accuracy and F1 are
# identical before and after. It matters here because the fusion
# module in the deployment folder pools the two modalities by
# their log-probabilities. An overconfident modality contributes
# a sharper log vector and wins arguments it has not earned.
# Calibrating both sides is what makes the comparison fair.
#
# Temperature scaling is the standard single-parameter fix
# (Guo et al., 2017, "On Calibration of Modern Neural
# Networks"): divide the logits by one scalar T fitted on the
# validation split by minimising NLL. One parameter cannot
# overfit, and because it is monotone it cannot change any
# prediction - only how confident that prediction claims to be.
# ============================================================

import numpy as np


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def expected_calibration_error(probabilities, labels, bins=15):
    """Average gap between confidence and accuracy, bucketed by confidence.

    Predictions are grouped into equal-width confidence bins. In a
    perfectly calibrated model, the predictions in the "70% confident"
    bin are correct 70% of the time. ECE is the weighted mean of
    |accuracy - confidence| across bins: 0 is perfect, and a typical
    uncalibrated classifier lands somewhere around 0.05 to 0.15.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0
    total = len(labels)

    if total == 0:
        return 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):

        # Left-open intervals so each prediction lands in exactly one
        # bin; the first bin also takes confidence exactly 0.
        in_bin = (confidence > lower) & (confidence <= upper)

        count = int(in_bin.sum())

        if count == 0:
            continue

        error += (count / total) * abs(
            correct[in_bin].mean() - confidence[in_bin].mean()
        )

    return float(error)


# ============================================================
# SOFTMAX / NLL
# ============================================================

def softmax(logits):
    """Numerically stable softmax over the last axis."""

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)

    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits, labels, temperature):
    """Mean NLL of the true classes after scaling by `temperature`."""

    probabilities = softmax(logits / temperature)

    chosen = probabilities[np.arange(len(labels)), labels]

    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


# ============================================================
# FIT
# ============================================================

def fit_temperature(logits, labels, low=0.05, high=10.0, iterations=60):
    """Find the temperature minimising validation NLL.

    Returns (temperature, ece_before, ece_after).

    Solved by golden-section search rather than gradient descent. The
    objective is one-dimensional and unimodal in T, the whole fit takes
    a few dozen evaluations of a cheap function, and this way the
    calibration step needs no optimiser, no learning rate and no torch
    graph - it operates on the logits the evaluation pass already
    produced.
    """

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    ece_before = expected_calibration_error(softmax(logits), labels)

    golden = (np.sqrt(5.0) - 1.0) / 2.0

    a, b = low, high

    c = b - golden * (b - a)
    d = a + golden * (b - a)

    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)

    for _ in range(iterations):

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = negative_log_likelihood(logits, labels, d)

        if abs(b - a) < 1e-4:
            break

    temperature = float((a + b) / 2.0)

    ece_after = expected_calibration_error(
        softmax(logits / temperature), labels
    )

    return temperature, ece_before, ece_after


# ============================================================
# LOADING
# ============================================================

def load_temperature(checkpoint_directory, default=1.0):
    """Read the fitted temperature saved next to a checkpoint.

    Returns `default` when the file is absent, so checkpoints trained
    before calibration existed - including the three MTEB ones already
    on disk - keep loading unchanged.
    """

    import json

    path = checkpoint_directory / "calibration.json"

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as handle:
        return float(json.load(handle).get("temperature", default))


# ============================================================
# MANUAL CHECK
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(0)

    # Synthesise an overconfident classifier: correct most of the time,
    # but with logits scaled up so the reported confidence overshoots.
    count, classes = 4000, 7

    labels = rng.integers(0, classes, size=count)

    logits = rng.normal(0.0, 1.0, size=(count, classes))

    # Push the true class up for 70% of rows, then inflate everything.
    hit = rng.random(count) < 0.70
    logits[np.arange(count)[hit], labels[hit]] += 3.0
    logits *= 2.5

    temperature, before, after = fit_temperature(logits, labels)

    accuracy = (logits.argmax(axis=1) == labels).mean()
    confidence_before = softmax(logits).max(axis=1).mean()
    confidence_after = softmax(logits / temperature).max(axis=1).mean()

    print(f"accuracy              {accuracy:.4f}")
    print(f"mean confidence       {confidence_before:.4f} -> {confidence_after:.4f}")
    print(f"expected cal. error   {before:.4f} -> {after:.4f}")
    print(f"fitted temperature    {temperature:.4f}")

    assert after < before, "calibration should reduce ECE"

    # Temperature scaling is monotone, so it must not move any argmax.
    assert np.array_equal(
        logits.argmax(axis=1), (logits / temperature).argmax(axis=1)
    ), "temperature scaling changed a prediction"

    print("\nok - ECE reduced and no prediction changed")
