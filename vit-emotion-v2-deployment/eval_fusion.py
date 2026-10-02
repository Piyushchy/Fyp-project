"""
Does fusing the two modalities actually beat using either one alone?

    python eval_fusion.py
    python eval_fusion.py --tune
    python eval_fusion.py --refresh-reliability

The honest problem, stated up front
-----------------------------------
There is no paired corpus here. Nobody recorded a person's face and
their typed message at the same instant with a single agreed label, so
a true multimodal accuracy number cannot be computed from what this
project has.

The earlier `Training_program_cycling.py` dealt with that by pairing an
arbitrary face with an arbitrary sentence that merely shared a label
(`i % len(files)`, line 120) and then *training* on the result. That is
not a valid dataset: it teaches the fusion layer that any face of class
c co-occurs with any sentence of class c, which is precisely the
assumption the layer was supposed to learn or reject.

This script does something different in kind. It pairs synthetically
too - it has to - but:

  * it only ever *evaluates*, never trains, so nothing can learn the
    artificial correspondence;
  * the mismatch rate is a controlled independent variable rather than
    an accident, so the question becomes "how does fusion degrade as
    the modalities stop agreeing", which is answerable from simulated
    pairs and is the question that actually matters for a live system;
  * both modalities' probability vectors come from the real trained
    models on real held-out inputs. Only the pairing is synthetic.

Read the output as a characterisation of the fusion rule, not as a
multimodal benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from fusion import FusionConfig, FusionEngine, TextEvidence
from labels import NUM_SHARED_LABELS, SHARED_LABELS

BASE_DIR = Path(__file__).resolve().parent
TEXT_MODULE_DIR = BASE_DIR.parent / "text-emotion-module"
# Per-model, matching eval_face.py's output. The old unsuffixed
# results_face_eval.json was written by a harness that ran the v2
# weights under v3's config against an eval set drawn from v3's own
# training data, so it described no model that has ever been served.
FACE_EVAL_PATH = BASE_DIR / (
    f"results_face_eval.{os.environ.get('EMOTION_MODEL', 'v4').lower()}.json"
)
RELIABILITY_PATH = BASE_DIR / "reliability.json"
RESULTS_PATH = BASE_DIR / "results_fusion_eval.json"


# ============================================================
# SYNTHETIC MODALITY OUTPUTS
# ============================================================
#
# When a real per-modality confusion matrix is available it is used to
# sample realistic vectors; otherwise the fallback below generates
# vectors with a specified accuracy and sharpness. Either way each
# modality gets its own error profile, because the entire point of
# reliability weighting is that they fail on different classes.
# ============================================================


def sample_from_confusion(
    true_index: int,
    confusion: np.ndarray,
    sharpness: float,
    confidence_gap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a plausible probability vector for a model with this confusion.

    The row of the confusion matrix gives how often this model calls a
    true class c each of the seven labels. A predicted label is drawn
    from that row, then turned into a distribution peaked on it. This
    reproduces the model's real error *pattern* - which classes it
    confuses with which - rather than assuming errors are uniform.

    confidence_gap is how many logits *less* peaked a wrong prediction
    is than a right one. It exists because a calibrated classifier is
    measurably less confident when it is wrong, and that correlation is
    the entire mechanism a confidence-weighted pool exploits: it is
    what lets the pool tell an earned vote from a guess.

    It is a swept parameter rather than a fixed assumption. At
    confidence_gap = 0 a model is equally confident right or wrong, no
    fusion rule can do better than picking the stronger modality, and
    the sweep will show exactly that. Both modalities here are
    temperature-calibrated (see calibration.py and the fitted
    temperature in emotion_config.json), so a positive gap is the
    realistic regime - but the zero-gap row is reported too, because a
    result that only holds under a favourable assumption should be
    shown next to the assumption.
    """

    row = confusion[true_index].astype(np.float64)

    total = row.sum()
    row = (
        row / total if total > 0
        else np.full(NUM_SHARED_LABELS, 1.0 / NUM_SHARED_LABELS)
    )

    predicted = int(rng.choice(NUM_SHARED_LABELS, p=row))

    peak = sharpness if predicted == true_index else sharpness - confidence_gap

    logits = rng.normal(0.0, 0.6, size=NUM_SHARED_LABELS)
    logits[predicted] += peak

    shifted = logits - logits.max()
    probabilities = np.exp(shifted)

    return probabilities / probabilities.sum()


def synthetic_confusion(accuracy: float, per_class_bias: np.ndarray) -> np.ndarray:
    """A confusion matrix with the requested mean accuracy.

    per_class_bias scales each class's diagonal, so a modality can be
    strong on some emotions and weak on others the way real ones are.
    """

    confusion = np.zeros((NUM_SHARED_LABELS, NUM_SHARED_LABELS))

    for index in range(NUM_SHARED_LABELS):

        correct = float(np.clip(accuracy * per_class_bias[index], 0.05, 0.97))

        confusion[index] = (1.0 - correct) / (NUM_SHARED_LABELS - 1)
        confusion[index, index] = correct

    return confusion


# ============================================================
# LOADING REAL PROFILES
# ============================================================


def load_face_profile() -> tuple[np.ndarray, np.ndarray, str]:
    """Per-class F1 and a confusion matrix for the face modality."""

    reliability = json.loads(RELIABILITY_PATH.read_text(encoding="utf-8"))

    per_class = np.array(
        [reliability["face"]["per_class_f1"][label] for label in SHARED_LABELS]
    )

    source = "reliability.json (README evaluation)"

    if FACE_EVAL_PATH.exists():
        payload = json.loads(FACE_EVAL_PATH.read_text(encoding="utf-8"))

        # The row that carries a stored logit_bias is the one server.py
        # serves. Taking the last row instead picked '+ calibration',
        # which is temperature-only - a configuration that differs from
        # the deployed one by 7 accuracy points and 35 points of
        # neutral F1, so fusion was being tuned against a face modality
        # far weaker than the one it actually receives.
        deployed = [
            row for row in payload["configurations"]
            if row.get("logit_bias") is not None
        ]

        best = deployed[-1] if deployed else payload["configurations"][-1]

        per_class = np.array(
            [best["per_class_f1"][label] for label in SHARED_LABELS]
        )
        source = f"{FACE_EVAL_PATH.name} ({best['name']})"

    confusion = synthetic_confusion(
        accuracy=float(per_class.mean()),
        per_class_bias=per_class / per_class.mean(),
    )

    return per_class, confusion, source


def load_text_profile() -> tuple[np.ndarray, np.ndarray, str]:
    """Per-class F1 and a confusion matrix for the text modality.

    Prefers the real GoEmotions benchmark; falls back to the placeholder
    in reliability.json when the model has not been benchmarked yet.
    """

    benchmark = TEXT_MODULE_DIR / "results" / "benchmark_goemotions.json"

    if benchmark.exists():

        payload = json.loads(benchmark.read_text(encoding="utf-8"))

        best = max(payload["results"], key=lambda r: r["metrics"]["macro_f1"])

        per_class = np.array([
            best["metrics"]["per_class"][label]["f1"] for label in SHARED_LABELS
        ])

        confusion = np.array(best["confusion_matrix"], dtype=np.float64)

        return per_class, confusion, f"benchmark_goemotions.json ({best['display_name']})"

    reliability = json.loads(RELIABILITY_PATH.read_text(encoding="utf-8"))

    per_class = np.array(
        [reliability["text"]["per_class_f1"][label] for label in SHARED_LABELS]
    )

    confusion = synthetic_confusion(
        accuracy=float(per_class.mean()),
        per_class_bias=per_class / per_class.mean(),
    )

    return per_class, confusion, "reliability.json (placeholder - train the model)"


# ============================================================
# THE SWEEP
# ============================================================


def run_sweep(
    engine: FusionEngine,
    face_confusion: np.ndarray,
    text_confusion: np.ndarray,
    mismatch_rates,
    trials: int,
    face_quality: float,
    text_age: float,
    token_count: int,
    face_sharpness: float,
    text_sharpness: float,
    confidence_gap: float,
    seed: int,
) -> list[dict]:
    """Accuracy of each strategy as the modalities are pulled apart.

    `mismatch` is the fraction of trials where the text talks about a
    different emotion than the face shows - the person typing "fine"
    through gritted teeth, or laughing while describing something sad.
    At 0 the modalities always refer to the same state; at 0.5 half the
    messages are about something else entirely.
    """

    rows = []

    for mismatch in mismatch_rates:

        # Independent streams per role. With one shared stream the
        # mismatch branch consumes a varying number of draws, which
        # desynchronises the face sequence and makes the face-only
        # column wander between rows - it would look as though the
        # mismatch rate affected the face, which it cannot.
        truth_rng = np.random.default_rng(seed)
        face_rng = np.random.default_rng(seed + 1)
        text_rng = np.random.default_rng(seed + 2)
        mismatch_rng = np.random.default_rng(seed + 3)

        face_hits = text_hits = fused_hits = 0
        conflicts = 0
        agreements = []

        for _ in range(trials):

            truth = int(truth_rng.integers(NUM_SHARED_LABELS))

            face_vector = sample_from_confusion(
                truth, face_confusion, face_sharpness, confidence_gap, face_rng
            )

            if mismatch_rng.random() < mismatch:
                # The text is about a different emotion. The ground
                # truth stays what the face is expressing, because in a
                # live session the face is the continuous signal and the
                # message is the intermittent one.
                other = int(mismatch_rng.integers(NUM_SHARED_LABELS - 1))
                text_truth = other + (1 if other >= truth else 0)
            else:
                text_truth = truth

            text_vector = sample_from_confusion(
                text_truth, text_confusion, text_sharpness,
                confidence_gap, text_rng,
            )

            evidence = TextEvidence(
                probabilities=text_vector,
                token_count=token_count,
                text="",
                timestamp=0.0,
                label=SHARED_LABELS[int(text_vector.argmax())],
                confidence=float(text_vector.max()),
            )

            result = engine.fuse(
                face_probabilities=face_vector,
                face_quality=face_quality,
                text_evidence=evidence,
                now=text_age,
            )

            face_hits += int(face_vector.argmax() == truth)
            text_hits += int(text_vector.argmax() == truth)
            fused_hits += int(result.label == SHARED_LABELS[truth])

            conflicts += int(result.conflicted)
            agreements.append(result.agreement)

        rows.append({
            "mismatch": mismatch,
            "face_only": face_hits / trials,
            "text_only": text_hits / trials,
            "fused": fused_hits / trials,
            "conflict_rate": conflicts / trials,
            "mean_agreement": float(np.mean(agreements)),
        })

    return rows


def print_sweep(rows: list[dict]) -> None:

    print()
    print("=" * 84)
    print("FUSION vs SINGLE MODALITY  (controlled-mismatch simulation)")
    print("=" * 84)
    print(
        f"{'mismatch':>9} {'face only':>11} {'text only':>11} {'fused':>9} "
        f"{'gain':>8} {'conflict':>10} {'agree':>8}"
    )
    print("-" * 84)

    for row in rows:

        best_single = max(row["face_only"], row["text_only"])
        gain = (row["fused"] - best_single) * 100

        print(
            f"{row['mismatch']*100:>8.0f}% "
            f"{row['face_only']*100:>10.2f}% "
            f"{row['text_only']*100:>10.2f}% "
            f"{row['fused']*100:>8.2f}% "
            f"{gain:>+7.2f} "
            f"{row['conflict_rate']*100:>9.1f}% "
            f"{row['mean_agreement']:>7.3f}"
        )

    print("-" * 84)
    print("gain = fused minus the better of the two single modalities.")
    print("=" * 84)


# ============================================================
# DEGRADATION CHECKS
# ============================================================


def run_degradation(engine: FusionEngine, face_confusion, text_confusion,
                    trials: int, confidence_gap: float, seed: int) -> None:
    """Confirm the weighting responds to quality and age as designed."""

    print("\n" + "=" * 84)
    print("DYNAMIC WEIGHTING  (matched modalities; one factor varied at a time)")
    print("=" * 84)

    def accuracy(face_quality: float, age: float, tokens: int) -> tuple[float, float, float]:

        truth_rng = np.random.default_rng(seed)
        face_rng = np.random.default_rng(seed + 1)
        text_rng = np.random.default_rng(seed + 2)

        hits = 0
        face_share = []
        text_share = []

        for _ in range(trials):
            truth = int(truth_rng.integers(NUM_SHARED_LABELS))
            face_vector = sample_from_confusion(
                truth, face_confusion, 2.0, confidence_gap, face_rng
            )
            text_vector = sample_from_confusion(
                truth, text_confusion, 2.0, confidence_gap, text_rng
            )

            evidence = TextEvidence(
                text_vector, tokens, "", 0.0,
                SHARED_LABELS[int(text_vector.argmax())], float(text_vector.max()),
            )

            result = engine.fuse(face_vector, face_quality, evidence, now=age)

            hits += int(result.label == SHARED_LABELS[truth])
            face_share.append(result.face.influence)
            text_share.append(result.text.influence)

        return hits / trials, float(np.mean(face_share)), float(np.mean(text_share))

    print("\nFace quality (text fixed: fresh, 12 tokens)")
    print(f"{'quality':>9} {'accuracy':>10} {'face share':>12} {'text share':>12}")
    for quality in (1.0, 0.75, 0.5, 0.25, 0.1):
        acc, fs, ts = accuracy(quality, 0.0, 12)
        print(f"{quality:>9.2f} {acc*100:>9.2f}% {fs:>11.3f} {ts:>11.3f}")

    print("\nText age (face fixed at quality 0.8)")
    print(f"{'age (s)':>9} {'accuracy':>10} {'face share':>12} {'text share':>12}")
    for age in (0, 15, 30, 60, 120, 200):
        acc, fs, ts = accuracy(0.8, float(age), 12)
        print(f"{age:>9d} {acc*100:>9.2f}% {fs:>11.3f} {ts:>11.3f}")

    print("\nMessage length (face fixed at quality 0.8, text fresh)")
    print(f"{'tokens':>9} {'accuracy':>10} {'face share':>12} {'text share':>12}")
    for tokens in (1, 2, 4, 8, 20):
        acc, fs, ts = accuracy(0.8, 0.0, tokens)
        print(f"{tokens:>9d} {acc*100:>9.2f}% {fs:>11.3f} {ts:>11.3f}")

    print("=" * 84)


# ============================================================
# CONFIDENCE-GAP SWEEP
# ============================================================


def run_confidence_sweep(
    engine: FusionEngine,
    face_confusion: np.ndarray,
    text_confusion: np.ndarray,
    trials: int,
    seed: int,
) -> list[dict]:
    """How much fusion depends on the modalities being calibrated.

    This is the sweep that decides whether the whole approach is sound.
    A logarithmic opinion pool can only beat its best single input if
    the inputs carry usable information about *when* they are wrong. If
    a model is exactly as confident when wrong as when right, there is
    nothing to arbitrate on and no weighting scheme can help.
    """

    gaps = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    rows = []

    for gap in gaps:

        sweep = run_sweep(
            engine, face_confusion, text_confusion,
            [0.0, 0.15], trials, 0.8, 0.0, 12, 2.0, 2.0, gap, seed,
        )

        clean, noisy = sweep[0], sweep[1]

        rows.append({
            "confidence_gap": gap,
            "face_only": clean["face_only"],
            "text_only": clean["text_only"],
            "fused_matched": clean["fused"],
            "fused_15pct_mismatch": noisy["fused"],
        })

    return rows


def print_confidence_sweep(rows: list[dict]) -> None:

    print()
    print("=" * 84)
    print("SENSITIVITY TO CALIBRATION")
    print("=" * 84)
    print(
        f"{'conf. gap':>10} {'face only':>11} {'text only':>11} "
        f"{'fused':>9} {'gain':>8} {'fused@15%':>11}"
    )
    print("-" * 84)

    for row in rows:

        best_single = max(row["face_only"], row["text_only"])
        gain = (row["fused_matched"] - best_single) * 100

        print(
            f"{row['confidence_gap']:>10.1f} "
            f"{row['face_only']*100:>10.2f}% "
            f"{row['text_only']*100:>10.2f}% "
            f"{row['fused_matched']*100:>8.2f}% "
            f"{gain:>+7.2f} "
            f"{row['fused_15pct_mismatch']*100:>10.2f}%"
        )

    print("-" * 84)
    print("confidence gap = how many logits less peaked a wrong prediction is")
    print("than a right one. 0 means confidence carries no information about")
    print("correctness, and no fusion rule can beat the better single modality.")
    print("=" * 84)


# ============================================================
# TUNING
# ============================================================


def tune_base_weights(face_confusion, text_confusion, trials: int,
                      confidence_gap: float, seed: int) -> dict:
    """Grid-search the two base weights against a realistic mismatch mix.

    Tuned against a spread of mismatch rates rather than at zero,
    because a system optimised purely for perfectly-agreeing modalities
    would lean on text far harder than a live session can support.
    """

    print("\nTuning base weights (this takes a moment) ...")

    candidates = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    mismatch_mix = [0.0, 0.1, 0.2, 0.35]

    best = None

    for face_base in candidates:
        for text_base in candidates:

            config = FusionConfig.load()
            config.face.base_weight = face_base
            config.text.base_weight = text_base

            engine = FusionEngine(config)

            rows = run_sweep(
                engine, face_confusion, text_confusion,
                mismatch_mix, trials, 0.8, 0.0, 12, 2.0, 2.0,
                confidence_gap, seed,
            )

            score = float(np.mean([row["fused"] for row in rows]))

            if best is None or score > best["score"]:
                best = {
                    "face_base": face_base,
                    "text_base": text_base,
                    "score": score,
                }

    print(
        f"best: face_base={best['face_base']}  text_base={best['text_base']}  "
        f"mean fused accuracy {best['score']*100:.2f}%"
    )

    return best


# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--face-quality", type=float, default=0.8)
    parser.add_argument("--text-age", type=float, default=0.0)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument(
        "--confidence-gap",
        type=float,
        default=1.5,
        help=(
            "How many logits less peaked a wrong prediction is than a right "
            "one. 0 models a model whose confidence says nothing about its "
            "correctness. See the calibration sensitivity table."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search the base weights and report the best pair.",
    )
    parser.add_argument(
        "--refresh-reliability",
        action="store_true",
        help="Write the measured per-class F1 of both modalities into reliability.json.",
    )

    arguments = parser.parse_args()

    face_f1, face_confusion, face_source = load_face_profile()
    text_f1, text_confusion, text_source = load_text_profile()

    print("Modality profiles")
    print(f"  face: {face_source}")
    print(f"  text: {text_source}")
    print()
    print(f"{'class':<10} {'face F1':>9} {'text F1':>9}")
    for index, label in enumerate(SHARED_LABELS):
        print(f"{label:<10} {face_f1[index]*100:>8.1f}% {text_f1[index]*100:>8.1f}%")
    print(f"{'macro':<10} {face_f1.mean()*100:>8.1f}% {text_f1.mean()*100:>8.1f}%")

    if arguments.refresh_reliability:

        reliability = json.loads(RELIABILITY_PATH.read_text(encoding="utf-8"))

        reliability["face"]["per_class_f1"] = {
            label: round(float(face_f1[index]), 4)
            for index, label in enumerate(SHARED_LABELS)
        }
        reliability["face"]["source"] = face_source

        reliability["text"]["per_class_f1"] = {
            label: round(float(text_f1[index]), 4)
            for index, label in enumerate(SHARED_LABELS)
        }
        reliability["text"]["source"] = text_source

        RELIABILITY_PATH.write_text(
            json.dumps(reliability, indent=2) + "\n", encoding="utf-8"
        )

        print(f"\nWrote measured per-class F1 to {RELIABILITY_PATH}")

    engine = FusionEngine()

    mismatch_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    confidence_rows = run_confidence_sweep(
        engine, face_confusion, text_confusion, arguments.trials, arguments.seed
    )

    print_confidence_sweep(confidence_rows)

    rows = run_sweep(
        engine, face_confusion, text_confusion, mismatch_rates,
        arguments.trials, arguments.face_quality, arguments.text_age,
        arguments.tokens, 2.0, 2.0, arguments.confidence_gap, arguments.seed,
    )

    print_sweep(rows)

    run_degradation(
        engine, face_confusion, text_confusion, arguments.trials,
        arguments.confidence_gap, arguments.seed,
    )

    tuned = None
    if arguments.tune:
        tuned = tune_base_weights(
            face_confusion, text_confusion, arguments.trials // 2,
            arguments.confidence_gap, arguments.seed,
        )

    payload = {
        "note": (
            "Controlled-mismatch simulation over per-modality confusion "
            "profiles. Pairing is synthetic; nothing is trained on it."
        ),
        "face_source": face_source,
        "text_source": text_source,
        "trials": arguments.trials,
        "face_quality": arguments.face_quality,
        "confidence_gap": arguments.confidence_gap,
        "calibration_sensitivity": confidence_rows,
        "sweep": rows,
        "tuned_base_weights": tuned,
    }

    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
