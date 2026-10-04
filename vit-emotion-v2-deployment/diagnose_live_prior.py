# ============================================================
# WHY THE LIVE SESSION OVER-CALLS DISGUST AND SURPRISE
# ============================================================
#
#   python diagnose_live_prior.py --data eval_data_unseen/test/webcam
#
# The complaint is that a webcam session shows 'disgust' and
# 'surprise' on faces that are plainly neutral or happy. The
# balanced test set does not reproduce that - on 350 images with
# 50 per class, disgust is UNDER-predicted 0.22x and surprise is
# only 1.16x. So the eval set says the model is roughly even, and
# the live session says it is not. Both are true, and the gap
# between them is the bug.
#
# The eval set has a uniform class prior: one face in seven is a
# disgust face. A webcam session does not. Someone sitting at a
# laptop is neutral most of the time, happy some of the time, and
# genuinely disgusted almost never. Call that split 55/20/.../2.
#
# A classifier trained and calibrated on a uniform prior carries
# that prior in its head. Bayes says the posterior it emits is
#
#     p(c | x)  ~  likelihood(x | c) * p_train(c)
#
# so when the real p(c) is nothing like p_train(c), every rare
# class is over-emitted by roughly the ratio between them. At a
# 2% true rate for disgust against a 14.3% assumed rate, that is
# a 7x inflation of its false positives - and because a session
# is mostly neutral frames, almost every disgust the user sees is
# a neutral face that leaked.
#
# This script measures that leak and tests the correction.
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_face as ef
from labels import SHARED_LABELS

BASE_DIR = Path(__file__).resolve().parent

NUM = len(SHARED_LABELS)
INDEX = {name: i for i, name in enumerate(SHARED_LABELS)}


# ============================================================
# WHAT A WEBCAM SESSION ACTUALLY LOOKS LIKE
# ============================================================
#
# Not measured from this project's users - there is no such
# recording - so it is an explicit assumption, written here where
# it can be argued with rather than buried in a constant. The
# shape is what matters: neutral dominates, happy is second,
# disgust and fear are near-absent. Any plausible session prior
# has that shape, and the conclusion holds for all of them.
# ============================================================

LIVE_PRIOR = {
    "angry": 0.04,
    "disgust": 0.02,
    "fear": 0.03,
    "happy": 0.20,
    "neutral": 0.55,
    "sad": 0.08,
    "surprise": 0.08,
}


def prior_vector(mapping: dict[str, float]) -> np.ndarray:
    vector = np.array([mapping[name] for name in SHARED_LABELS])
    return vector / vector.sum()


# ============================================================
# COLLECTING LOGITS ONCE
# ============================================================

def collect(samples, detector, classifier, config) -> tuple[np.ndarray, np.ndarray]:
    """(logits, labels) for every detected face, raw and uncalibrated."""

    result = ef.run_configuration(
        ef.Configuration(
            "collect", ef.CROP_MODE_DEPLOYED, ef.USE_TTA_DEPLOYED,
            1.0, "raw logits", logit_bias=None,
            multicrop=ef.USE_MULTICROP_DEPLOYED,
        ),
        samples, detector, classifier,
    )

    logits = []
    labels = []

    for entry, label in zip(result["_logits"], result["_labels"]):
        if entry is None:
            continue

        logits.append(np.asarray(entry, dtype=np.float64))
        labels.append(int(label))

    return np.array(logits), np.array(labels)


# ============================================================
# THE SIMULATION
# ============================================================
#
# The test set is balanced, so it cannot be replayed directly as a
# session. Instead each true class is WEIGHTED by how often it
# would really occur. That reuses every measured image and makes
# no assumption about the model - only about how often a user is
# actually disgusted.
# ============================================================

def session_distribution(
    logits: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Expected predicted-class shares over a realistic session."""

    predictions = np.argmax(logits + offset, axis=1)

    shares = np.zeros(NUM)
    correct = 0.0

    for true_index in range(NUM):
        mask = labels == true_index

        if not mask.any():
            continue

        # How this class's faces get labelled, as a distribution.
        counts = np.bincount(predictions[mask], minlength=NUM).astype(float)
        counts /= counts.sum()

        shares += prior[true_index] * counts
        correct += prior[true_index] * counts[true_index]

    return shares, float(correct)


def show(title: str, shares: np.ndarray, accuracy: float, prior: np.ndarray) -> None:
    print(f"\n  {title}")
    print(f"    {'class':<10} {'true %':>8} {'shown %':>9} {'inflation':>10}")

    for index, name in enumerate(SHARED_LABELS):
        inflation = shares[index] / max(prior[index], 1e-9)

        flag = ""

        if inflation >= 2.0:
            flag = "  <--"

        print(
            f"    {name:<10} {prior[index] * 100:>7.1f}% "
            f"{shares[index] * 100:>8.1f}% {inflation:>9.2f}x{flag}"
        )

    print(f"    session accuracy {accuracy:.4f}")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args()

    samples = ef.load_index(arguments.data, None)
    classifier, config = ef.build_classifier(prefer_gpu=not arguments.cpu)
    detector = ef.build_detector()

    temperature = float(config.get("temperature", 1.0))
    stored_bias = np.asarray(
        config.get("logit_bias") or np.zeros(NUM), dtype=np.float64
    )

    print(f"\n{len(samples)} images, model {ef.ONNX_PATH.name}")

    logits, labels = collect(samples, detector, classifier, config)

    print(f"{len(labels)} faces detected")

    # Everything below compares argmax under different offsets, so
    # temperature is folded in once here.
    scaled = logits / temperature

    prior = prior_vector(LIVE_PRIOR)

    # The prior the calibration was fitted against: the eval set's own
    # class balance, which is uniform by construction.
    fitted_prior = np.bincount(labels, minlength=NUM).astype(float)
    fitted_prior /= fitted_prior.sum()

    print("\n" + "=" * 68)
    print("WHAT A REALISTIC SESSION WOULD SHOW")
    print("=" * 68)

    results = {}

    # ----------------------------------------------------
    # 1. As deployed today
    # ----------------------------------------------------

    shares, accuracy = session_distribution(scaled, labels, prior, stored_bias)
    show("as deployed (stored logit bias)", shares, accuracy, prior)
    results["deployed"] = {"shares": shares.tolist(), "accuracy": accuracy}

    # ----------------------------------------------------
    # 2. Prior-corrected
    # ----------------------------------------------------
    #
    # Add log(p_live / p_fitted) to the logits. This is the exact
    # Bayes correction for running a classifier under a prior
    # different from the one it was calibrated on, and it costs one
    # addition per frame.
    # ----------------------------------------------------

    correction = np.log(np.clip(prior, 1e-6, None)) - np.log(
        np.clip(fitted_prior, 1e-6, None)
    )

    shares_pc, accuracy_pc = session_distribution(
        scaled, labels, prior, stored_bias + correction
    )
    show("with prior correction", shares_pc, accuracy_pc, prior)

    results["prior_corrected"] = {
        "shares": shares_pc.tolist(),
        "accuracy": accuracy_pc,
        "correction": correction.tolist(),
    }

    # ----------------------------------------------------
    # 3. Half-strength prior correction
    # ----------------------------------------------------
    #
    # The full correction optimises for the assumed prior. If that
    # assumption is wrong the model goes deaf to real expressions -
    # it would need overwhelming evidence to ever say 'disgust'.
    # Halving it in log space is the standard hedge.
    # ----------------------------------------------------

    shares_h, accuracy_h = session_distribution(
        scaled, labels, prior, stored_bias + 0.5 * correction
    )
    show("with HALF prior correction", shares_h, accuracy_h, prior)

    results["half_prior_corrected"] = {
        "shares": shares_h.tolist(),
        "accuracy": accuracy_h,
        "correction": (0.5 * correction).tolist(),
    }

    # ----------------------------------------------------
    # Balanced-set cost of the same change
    # ----------------------------------------------------
    #
    # A correction that helps the session is worthless if it wrecks
    # the benchmark, so the same offsets are scored on the uniform
    # test set too.
    # ----------------------------------------------------

    print("\n" + "=" * 68)
    print("COST ON THE BALANCED TEST SET")
    print("=" * 68)
    print(f"  {'offset':<28} {'accuracy':>9} {'macro-F1':>9}")

    for name, offset in (
        ("as deployed", stored_bias),
        ("+ full prior correction", stored_bias + correction),
        ("+ half prior correction", stored_bias + 0.5 * correction),
    ):
        predictions = np.argmax(scaled + offset, axis=1)
        accuracy_b = float((predictions == labels).mean())
        macro, _ = ef.macro_f1(labels, predictions)

        print(f"  {name:<28} {accuracy_b:>9.4f} {macro:>9.4f}")

        results.setdefault("balanced", {})[name] = {
            "accuracy": accuracy_b, "macro_f1": float(macro),
        }

    output = BASE_DIR / "results_live_prior.json"

    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": ef.ONNX_PATH.name,
                "live_prior": LIVE_PRIOR,
                "fitted_prior": fitted_prior.tolist(),
                "results": results,
            },
            handle, indent=2,
        )

    print(f"\nwrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
