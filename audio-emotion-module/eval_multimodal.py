# ============================================================
# MULTIMODAL EVALUATION
# ============================================================
#
# Scores the face + audio + text fusion on the MELD test split
# and writes results/multimodal.json.
#
#   python eval_multimodal.py
#
# Why MELD and not the other corpora
# ----------------------------------
# MELD ships a transcript per clip, so its test split is the one
# place in this project where two real modalities describe the
# SAME utterance. Everywhere else the only way to evaluate a
# pool is to pair a face prediction with a text prediction drawn
# independently - which is what eval_fusion.py has to do, and
# which quietly assumes the two modalities fail independently.
# They do not: a quiet, flat delivery and a short, affectless
# sentence tend to co-occur, and independent pairing hides that
# correlation entirely. Measured on real pairs the fusion gain
# is smaller and the number is worth more.
#
# What is real here and what is not
# ---------------------------------
#   audio   REAL. The trained encoder's own probabilities on
#           these exact clips, read from the checkpoint's saved
#           test predictions.
#   text    REAL. The deployed TinyBERT/GoEmotions checkpoint
#           run over each clip's transcript.
#   face    SIMULATED. MELD is audio and text only - there are no
#           aligned face crops for these utterances. Face vectors
#           are drawn from the deployed face model's MEASURED
#           confusion matrix, conditioned on the true label, so
#           the error *pattern* is the real one even though the
#           individual frames are not.
#
# That asymmetry is the main limitation of this script and it is
# stated in the output rather than buried here: any face number
# below is a model of the face modality, not a measurement of
# it. The audio-text pair, which is the part this module adds,
# is fully real.
# ============================================================

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from config import DATA_DIR, LABELS, MODULE_DIR, RESULTS_DIR, checkpoint_path
from data import load_manifest
from metrics import compute_metrics, expected_calibration_error

DEPLOYMENT_DIR = MODULE_DIR.parent / "vit-emotion-v2-deployment"
TEXT_MODULE_DIR = MODULE_DIR.parent / "text-emotion-module"

sys.path.insert(0, str(DEPLOYMENT_DIR))

from fusion import (  # noqa: E402
    AudioEvidence,
    FusionConfig,
    FusionEngine,
    TextEvidence,
)

RESULTS_PATH = RESULTS_DIR / "multimodal.json"

SEED = 42


# ============================================================
# CROSS-MODULE IMPORTS
# ============================================================
#
# The audio and text modules both name their files config.py,
# data.py, metrics.py and modeling.py. Python's top-level module
# namespace is flat, so whichever imports first owns the name -
# and this script has already imported the audio module's. A
# plain `from predict import ...` then hands the text module the
# AUDIO config and dies on the first symbol it wants.
#
# Neither module is a package, and making them into one would
# change every import in both plus the way they are invoked from
# the command line. Parking the colliding names for the duration
# of the import is the smaller change.
# ============================================================

COLLIDING_MODULES = (
    "config", "data", "metrics", "modeling", "calibration", "predict",
)


@contextlib.contextmanager
def text_module_namespace():
    """Let the text module's files import each other, then undo it."""

    parked = {
        name: sys.modules.pop(name)
        for name in COLLIDING_MODULES
        if name in sys.modules
    }

    saved_path = list(sys.path)
    sys.path.insert(0, str(TEXT_MODULE_DIR))

    try:
        yield
    finally:
        sys.path[:] = saved_path

        for name in COLLIDING_MODULES:
            sys.modules.pop(name, None)

        sys.modules.update(parked)


# ============================================================
# THE SIMULATED FACE LEG
# ============================================================

def face_confusion_matrix() -> tuple[np.ndarray, str]:
    """Row-normalised confusion for the DEPLOYED face model.

    Reuses eval_fusion.py's loader so the face modality is modelled
    identically in both evaluations; a second, slightly different
    model of the same thing would make the two sets of numbers
    incomparable for no reason.
    """

    import eval_fusion

    _, confusion, source = eval_fusion.load_face_profile()

    return confusion, source


def sample_face(
    true_index: int,
    confusion: np.ndarray,
    sharpness: float,
    confidence_gap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    import eval_fusion

    return eval_fusion.sample_from_confusion(
        true_index, confusion, sharpness, confidence_gap, rng
    )


# ============================================================
# THE REAL LEGS
# ============================================================

def load_audio_predictions(model_key: str) -> dict[str, np.ndarray]:
    """{uid: probability vector} from a trained checkpoint."""

    path = checkpoint_path(model_key) / "test_predictions.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train the audio model first:\n"
            f"  python train_audio.py --model {model_key}"
        )

    payload = np.load(path, allow_pickle=False)

    return {
        str(uid): probabilities
        for uid, probabilities in zip(payload["uids"], payload["probabilities"])
    }


def audio_probabilities_for(
    rows: list[dict], model_key: str, cache: Path
) -> dict[str, np.ndarray]:
    """Run a trained audio checkpoint over arbitrary manifest rows.

    train_audio.py only saves its predictions for the test split, but
    the base weights have to be tuned somewhere that is not the test
    split. This re-runs the encoder over the validation rows.
    """

    if cache.exists():
        payload = np.load(cache, allow_pickle=False)
        stored = {
            str(uid): vector
            for uid, vector in zip(payload["uids"], payload["probabilities"])
        }

        if all(row["uid"] in stored for row in rows):
            return stored

    import torch

    import modeling
    from config import MODELS
    from data import MfccSplit, WaveformSplit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = modeling.build(model_key).to(device).eval()
    model.load_state_dict(
        torch.load(checkpoint_path(model_key) / "model.pt", map_location=device)
    )

    kind = MODELS[model_key]["kind"]

    if kind == "ssl":
        dataset = WaveformSplit(rows)
    else:
        dataset = MfccSplit(rows, "tune")

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    outputs = []

    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device))
            outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    probabilities = np.concatenate(outputs)
    uids = [row["uid"] for row in rows]

    np.savez_compressed(
        cache, uids=np.array(uids), probabilities=probabilities
    )

    return dict(zip(uids, probabilities))


def load_text_predictions(
    rows: list[dict], model_key: str, cache: Path
) -> dict[str, np.ndarray]:
    """{uid: shared-space vector} for each row's transcript.

    Cached: the checkpoint is deterministic and this is the slowest
    step in the script, so a re-run with different fusion settings
    should not pay for it again.
    """

    if cache.exists():
        payload = np.load(cache, allow_pickle=False)
        stored = {
            str(uid): vector
            for uid, vector in zip(payload["uids"], payload["probabilities"])
        }

        if all(row["uid"] in stored for row in rows):
            return stored

    vectors = {}
    latencies = []

    # Scoring happens inside the context, not just the import: the
    # classifier reaches back into its own config on each call.
    with text_module_namespace():
        from predict import TextEmotionClassifier

        classifier = TextEmotionClassifier(model_key=model_key, device="cpu")

        for position, row in enumerate(rows):
            prediction = classifier.predict(row["transcript"])

            vectors[row["uid"]] = np.array(
                [prediction["shared_distribution"][label] for label in LABELS],
                dtype=np.float64,
            )

            latencies.append(prediction["latency_ms"])

            if (position + 1) % 500 == 0:
                print(f"    text {position + 1}/{len(rows)}", flush=True)

    uids = [row["uid"] for row in rows]

    np.savez_compressed(
        cache,
        uids=np.array(uids),
        probabilities=np.stack([vectors[uid] for uid in uids]),
        latency_ms=np.array(latencies),
    )

    return vectors


def token_counts(rows: list[dict]) -> dict[str, int]:
    """Whitespace token count per transcript.

    Fusion's informativeness factor wants the message length. The
    exact tokenizer count is in the text module, but it is within a
    token or two of this for conversational English and this avoids
    reloading the tokenizer when the predictions came from cache.
    """

    return {
        row["uid"]: len(row["transcript"].split()) for row in rows
    }


# ============================================================
# ONE CONFIGURATION
# ============================================================

def run_configuration(
    engine: FusionEngine,
    rows: list[dict],
    face: dict[str, np.ndarray],
    audio: dict[str, np.ndarray],
    text: dict[str, np.ndarray],
    tokens: dict[str, int],
    use_face: bool,
    use_audio: bool,
    use_text: bool,
    face_quality: float = 0.85,
    voiced_scale: float = 1.0,
    text_age: float = 2.0,
) -> dict:
    """Fuse every row under one on/off pattern and score the result."""

    truth = []
    probabilities = []
    conflicts = []
    modes = []

    started = time.perf_counter()

    for row in rows:
        uid = row["uid"]

        evidence_text = None

        if use_text:
            evidence_text = TextEvidence(
                probabilities=text[uid],
                token_count=tokens[uid],
                text=row["transcript"],
                timestamp=0.0,
                label="",
                confidence=float(text[uid].max()),
            )

        evidence_audio = None

        if use_audio:
            evidence_audio = AudioEvidence(
                probabilities=audio[uid],
                # The prepare step measured this per clip, so the
                # quality factor is the real voiced fraction of the
                # real window rather than a flat assumption.
                # voiced_scale attenuates it for the degradation
                # sweep: 1.0 is the measured value, 0.0 is silence.
                voiced_ratio=voiced_scale
                * min(1.0, row["voiced_seconds"] / max(1e-6, row["duration"])),
                timestamp=0.0,
            )

        result = engine.fuse(
            face[uid] if use_face else None,
            face_quality,
            evidence_text,
            now=text_age,
            audio_evidence=evidence_audio,
        )

        if result.probabilities is None:
            continue

        truth.append(row["label_id"])
        probabilities.append(result.probabilities)
        conflicts.append(result.conflicted)
        modes.append(result.mode)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not truth:
        return {"support": 0}

    truth = np.array(truth)
    probabilities = np.stack(probabilities)
    predictions = probabilities.argmax(axis=1)

    scores = compute_metrics(truth, predictions, probabilities)

    conflicts = np.array(conflicts)
    correct = predictions == truth

    scores["conflict_rate"] = float(conflicts.mean())

    # Accuracy split by whether the modalities disagreed. If the flag
    # is doing its job these two numbers are far apart, and the gap is
    # what justifies surfacing it in the UI at all.
    scores["accuracy_when_conflicted"] = (
        float(correct[conflicts].mean()) if conflicts.any() else None
    )
    scores["accuracy_when_agreed"] = (
        float(correct[~conflicts].mean()) if (~conflicts).any() else None
    )

    scores["fusion_ms_per_frame"] = elapsed_ms / len(truth)

    return scores


# ============================================================
# THE SWEEPS
# ============================================================

def degradation_sweep(
    engine, rows, face, audio, text, tokens, axis: str
) -> list[dict]:
    """Walk one quality axis from healthy to absent."""

    sweep = []

    if axis in ("face_quality", "voiced_scale"):
        values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    else:
        # Text age, in seconds. The last value is past the cutoff, so
        # it measures the collapse to face+audio rather than a decay.
        values = [0.0, 15.0, 30.0, 60.0, 120.0, 200.0]

    for value in values:
        kwargs = {"face_quality": 0.85, "voiced_scale": 1.0, "text_age": 2.0}
        kwargs[axis] = value

        scores = run_configuration(
            engine, rows, face, audio, text, tokens,
            use_face=True, use_audio=True, use_text=True,
            **kwargs,
        )

        sweep.append(
            {
                axis: value,
                "accuracy": scores["accuracy"],
                "macro_f1": scores["macro_f1"],
                "ece": scores["ece"],
            }
        )

    return sweep


# ============================================================
# BASE WEIGHT TUNING
# ============================================================

def tune_base_weights(
    engine: FusionEngine,
    rows: list[dict],
    face: dict[str, np.ndarray],
    audio: dict[str, np.ndarray],
    text: dict[str, np.ndarray],
    tokens: dict[str, int],
    trials: int,
    rng: np.random.Generator,
) -> dict:
    """Search the three base weights on the VALIDATION rows.

    Audio enters reliability.json at a weight nobody has measured.
    Leaving it at 1.0 would give a modality that scores in the
    thirties the same say as one that scores in the fifties, which
    understates the pool and would make the fusion look worse than
    it is for a reason that has nothing to do with the fusion.

    Tuned on validation and never on test. The whole point of the
    held-out split is lost if the pool's own parameters have seen it,
    and a weight set chosen on the test rows would inflate every
    number this script reports.
    """

    original = (
        engine.config.face.base_weight,
        engine.config.audio.base_weight,
        engine.config.text.base_weight,
    )

    def score_with(face_weight, audio_weight, text_weight) -> float:
        engine.config.face.base_weight = face_weight
        engine.config.audio.base_weight = audio_weight
        engine.config.text.base_weight = text_weight

        scores = run_configuration(
            engine, rows, face, audio, text, tokens,
            use_face=True, use_audio=True, use_text=True,
        )

        return scores["macro_f1"]

    best = {
        "face": original[0],
        "audio": original[1],
        "text": original[2],
        "macro_f1": score_with(*original),
    }

    baseline = best["macro_f1"]

    for _ in range(trials):
        candidate = rng.uniform(0.1, 1.5, size=3)

        value = score_with(*candidate)

        if value > best["macro_f1"]:
            best = {
                "face": float(candidate[0]),
                "audio": float(candidate[1]),
                "text": float(candidate[2]),
                "macro_f1": value,
            }

    engine.config.face.base_weight = best["face"]
    engine.config.audio.base_weight = best["audio"]
    engine.config.text.base_weight = best["text"]

    best["baseline_macro_f1"] = baseline
    best["trials"] = trials
    best["tuned_on"] = f"{len(rows)} MELD validation clips"

    return best


# ============================================================
# THE COMPOSITE
# ============================================================

def composite_score(report: dict) -> dict:
    """A single 0-100 figure, from five components with stated weights.

    A composite is only honest if the recipe is visible, so every
    component, its raw value, its normalisation and its weight are
    returned alongside the total. Nothing here is a standard metric -
    it is this project's own summary of "is the fusion worth having",
    and the components are what that question decomposes into.
    """

    configurations = report["configurations"]

    trimodal = configurations["face+audio+text"]

    unimodal = [
        configurations[name]["macro_f1"]
        for name in ("face", "audio", "text")
    ]

    best_unimodal = max(unimodal)

    # 1. Raw capability. Macro-F1 of the full pool, against a ceiling
    #    of 0.70 - roughly what a strong supervised system reaches on
    #    seven-class MELD-domain emotion, so 100 means "at the state of
    #    the art", not "perfect".
    capability = min(1.0, trimodal["macro_f1"] / 0.70)

    # 2. Fusion gain. How much the pool beats the best single modality.
    #    Normalised against a 10-point macro-F1 improvement, which is a
    #    large gain for late fusion on real pairs.
    gain = float(np.clip((trimodal["macro_f1"] - best_unimodal) / 0.10, 0.0, 1.0))

    # 3. Calibration. 1 - ECE, floored at 0. A fused probability that
    #    is used for anything other than its argmax has to be
    #    trustworthy as a number.
    calibration = float(np.clip(1.0 - trimodal["ece"] / 0.25, 0.0, 1.0))

    # 4. Robustness. The worst macro-F1 across the three
    #    single-modality-loss configurations, as a fraction of the full
    #    pool's. A system that collapses when the microphone is muted
    #    is not a multimodal system.
    degraded = [
        configurations[name]["macro_f1"]
        for name in ("face+audio", "face+text", "audio+text")
    ]

    robustness = float(
        np.clip(min(degraded) / max(1e-6, trimodal["macro_f1"]), 0.0, 1.0)
    )

    # 5. Conflict discrimination. The gap between accuracy when the
    #    modalities agree and when they do not, normalised against 30
    #    points. This measures whether the conflict flag predicts
    #    anything - a flag that fires at random is worse than none.
    agreed = trimodal.get("accuracy_when_agreed")
    conflicted = trimodal.get("accuracy_when_conflicted")

    if agreed is None or conflicted is None:
        discrimination = 0.0
    else:
        discrimination = float(np.clip((agreed - conflicted) / 0.30, 0.0, 1.0))

    components = {
        "capability": {
            "raw": trimodal["macro_f1"],
            "normalised": capability,
            "weight": 0.35,
            "basis": "trimodal macro-F1 against a 0.70 ceiling",
        },
        "fusion_gain": {
            "raw": trimodal["macro_f1"] - best_unimodal,
            "normalised": gain,
            "weight": 0.25,
            "basis": "macro-F1 over the best unimodal, against +0.10",
        },
        "calibration": {
            "raw": trimodal["ece"],
            "normalised": calibration,
            "weight": 0.15,
            "basis": "1 - ECE/0.25",
        },
        "robustness": {
            "raw": min(degraded),
            "normalised": robustness,
            "weight": 0.15,
            "basis": "worst two-modality macro-F1 / trimodal macro-F1",
        },
        "conflict_discrimination": {
            "raw": (agreed - conflicted) if agreed and conflicted else 0.0,
            "normalised": discrimination,
            "weight": 0.10,
            "basis": "agreed minus conflicted accuracy, against 0.30",
        },
    }

    total = sum(
        entry["normalised"] * entry["weight"] for entry in components.values()
    )

    return {
        "score": round(100.0 * total, 1),
        "components": components,
        "best_unimodal_macro_f1": best_unimodal,
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-model", default="wavlm-base-plus")
    parser.add_argument("--text-model", default="tinybert")
    parser.add_argument("--sharpness", type=float, default=3.0)
    parser.add_argument("--confidence-gap", type=float, default=1.0)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="search base weights on the MELD validation split first",
    )
    parser.add_argument("--tune-trials", type=int, default=60)
    arguments = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("MULTIMODAL EVALUATION")
    print("=" * 62)

    manifest = load_manifest()

    def meld_rows(split: str) -> list[dict]:
        return [
            row
            for row in manifest
            if row["split"] == split
            and row["source"] == "meld"
            and row["transcript"].strip()
        ]

    rows = meld_rows("test")
    validation = meld_rows("validation")

    print(f"  rows            {len(rows)} MELD test clips with transcripts")

    audio = load_audio_predictions(arguments.audio_model)

    missing = [row["uid"] for row in rows if row["uid"] not in audio]

    if missing:
        raise SystemExit(
            f"{len(missing)} rows have no audio prediction; the checkpoint "
            f"was trained on a different manifest. Retrain or rebuild."
        )

    print(f"  audio           {arguments.audio_model} (real)")

    print(f"  text            {arguments.text_model} (real), scoring transcripts")

    text = load_text_predictions(
        rows, arguments.text_model, DATA_DIR / "text_predictions.npz"
    )

    confusion, face_source = face_confusion_matrix()

    print(f"  face            SIMULATED from {face_source}")

    rng = np.random.default_rng(SEED)

    face = {
        row["uid"]: sample_face(
            row["label_id"],
            confusion,
            arguments.sharpness,
            arguments.confidence_gap,
            rng,
        )
        for row in rows
    }

    tokens = token_counts(rows)

    engine = FusionEngine(FusionConfig.load())

    tuning = None

    if arguments.tune:
        print(f"\n--- tuning base weights on {len(validation)} validation clips ---")

        validation_audio = audio_probabilities_for(
            validation, arguments.audio_model,
            DATA_DIR / f"audio_validation_{arguments.audio_model}.npz",
        )

        validation_text = load_text_predictions(
            validation, arguments.text_model,
            DATA_DIR / "text_predictions_validation.npz",
        )

        validation_face = {
            row["uid"]: sample_face(
                row["label_id"], confusion,
                arguments.sharpness, arguments.confidence_gap, rng,
            )
            for row in validation
        }

        tuning = tune_base_weights(
            engine, validation, validation_face, validation_audio,
            validation_text, token_counts(validation),
            arguments.tune_trials, rng,
        )

        print(
            f"  face {tuning['face']:.3f}  audio {tuning['audio']:.3f}  "
            f"text {tuning['text']:.3f}"
        )
        print(
            f"  validation macro-F1 {tuning['baseline_macro_f1']:.4f} "
            f"-> {tuning['macro_f1']:.4f}"
        )

    patterns = {
        "face": (True, False, False),
        "audio": (False, True, False),
        "text": (False, False, True),
        "face+audio": (True, True, False),
        "face+text": (True, False, True),
        "audio+text": (False, True, True),
        "face+audio+text": (True, True, True),
    }

    configurations = {}

    print("\n--- configurations ---")
    print(
        f"  {'modalities':<18} {'acc':>7} {'macroF1':>8} {'wtdF1':>7} "
        f"{'ECE':>7} {'conflict':>9}"
    )

    for name, (use_face, use_audio, use_text) in patterns.items():
        scores = run_configuration(
            engine, rows, face, audio, text, tokens,
            use_face=use_face, use_audio=use_audio, use_text=use_text,
        )

        configurations[name] = scores

        print(
            f"  {name:<18} {scores['accuracy']:>7.4f} "
            f"{scores['macro_f1']:>8.4f} {scores['weighted_f1']:>7.4f} "
            f"{scores['ece']:>7.4f} {scores['conflict_rate']:>9.4f}"
        )

    print("\n--- degradation ---")

    sweeps = {}

    for axis in ("face_quality", "voiced_scale", "text_age"):
        sweeps[axis] = degradation_sweep(
            engine, rows, face, audio, text, tokens, axis
        )

        print(f"\n  {axis}")
        print(f"    {'value':>8} {'acc':>8} {'macroF1':>9}")

        for entry in sweeps[axis]:
            print(
                f"    {entry[axis]:>8.2f} {entry['accuracy']:>8.4f} "
                f"{entry['macro_f1']:>9.4f}"
            )

    report = {
        "dataset": "MELD test split, audio + transcript pairs",
        "rows": len(rows),
        "audio_model": arguments.audio_model,
        "text_model": arguments.text_model,
        "face_source": face_source,
        "face_is_simulated": True,
        "face_simulation": {
            "sharpness": arguments.sharpness,
            "confidence_gap": arguments.confidence_gap,
            "seed": SEED,
        },
        "labels": list(LABELS),
        "configurations": configurations,
        "degradation": sweeps,
        "base_weights": {
            "face": engine.config.face.base_weight,
            "audio": engine.config.audio.base_weight,
            "text": engine.config.text.base_weight,
        },
        "tuning": tuning,
    }

    report["composite"] = composite_score(report)

    print("\n" + "=" * 62)
    print(f"MULTIMODAL SCORE: {report['composite']['score']} / 100")
    print("=" * 62)

    for name, entry in report["composite"]["components"].items():
        print(
            f"  {name:<24} {entry['normalised']:.3f} "
            f"x {entry['weight']:.2f}   ({entry['basis']})"
        )

    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nwrote {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
