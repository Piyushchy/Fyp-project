"""
Measure what each inference-time change is actually worth.

    python eval_face.py --data <dir>            # one row per configuration
    python eval_face.py --data <dir> --sweep-crop
    python eval_face.py --data <dir> --roll-test
    python eval_face.py --data <dir> --fit-temperature

Expects a directory of labelled face images, one subdirectory per class,
named with the shared labels:

    <dir>/angry/*.jpg   <dir>/disgust/*.png   ... <dir>/surprise/*.jpg

FER2013's test split exported as folders is the obvious source. Anything
with the same layout works; unknown subdirectory names are skipped with a
warning rather than silently folded into another class.

Why this exists
---------------
A headline accuracy figure says nothing about what the deployment
actually does to a webcam frame: detect a face, crop it, hand the crop
to the model. The crop is the part under our control, so the crop is
what has to be measured. Every configuration below runs the same
weights over the same images - only the geometry and the
post-processing differ.

Choose --data by FRAMING, not by content. This is the opposite of what
this docstring used to say, and the old advice produced a wrong answer.

The rule it gave was "use the model's own distribution when deciding
anything about preprocessing", meaning FER2013. But FER2013 images are
*already* tight face crops: no background, no neck, the face filling
the frame. Run a detector over one and re-frame it and you are moving
it away from its native framing, so the crop that changes least wins
almost by construction. Measured on de-duplicated FER2013 test, v4
scores:

    box + 25%        66.61% acc   62.58% macro-F1
    aligned          62.65% acc   56.99% macro-F1

and on faces composited into 640x480 frames with background - which is
what a webcam actually produces - the ordering reverses:

    box + 25%        44.00% acc   38.69% macro-F1   21.58% neutral F1
    aligned          44.00% acc   40.55% macro-F1   32.77% neutral F1

Neither set is "wrong". They answer different questions. Preprocessing
decisions belong to whichever set has the same *geometry* as
deployment, because preprocessing is geometry - and a webcam hands the
model a head in a room, not a pre-cropped portrait.

--roll-test answers the follow-up question: alignment corrects head
roll, so it should only earn its resampling cost once there is roll to
correct. It applies a known rotation and reports where the sign flips.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from face_pipeline import (
    EYE_DISTANCE_RATIO,
    LEGACY_MARGIN,
    MULTICROP_RATIOS,
    ROLL_ALIGN_THRESHOLD_DEGREES,
    CropMode,
    EmotionClassifier,
    ProbabilitySmoother,
    assess_quality,
    build_detector,
    crop_face,
    describe_providers,
    select_providers,
    softmax,
)
from labels import NUM_SHARED_LABELS, SHARED_LABELS, assert_matches_config

BASE_DIR = Path(__file__).resolve().parent

# Model and config travel together. They were previously hardcoded to
# vit-emotion-v2.onnx *and* emotion_config.json - that is v2's weights
# paired with v3's config, so --fit-temperature fitted on v2 and wrote
# the result into the file the server reads for v3. It also meant v3,
# the shipped default, was never measured by this harness at all.
#
# EMOTION_MODEL matches server.py so the two cannot drift.
MODELS = {
    "v2": ("vit-emotion-v2.onnx", "emotion_config.v2.json"),
    "v3": ("vit-emotion-v3.onnx", "emotion_config.json"),
    "v4": ("vit-emotion-v4.onnx", "emotion_config.v4.json"),
    "v5": ("vit-emotion-v5.onnx", "emotion_config.v5.json"),
    "v6": ("vit-emotion-v6.onnx", "emotion_config.v6.json"),
}

MODEL_KEY = os.environ.get("EMOTION_MODEL", "v4").lower()
if MODEL_KEY not in MODELS:
    raise SystemExit(f"EMOTION_MODEL must be one of {sorted(MODELS)}")

# Mirror server.USE_TTA and server.CROP_MODE, so the "deployed" row and
# the calibration fit both match what is actually served.
USE_TTA_DEPLOYED = False
CROP_MODE_DEPLOYED = CropMode.ALIGNED
USE_MULTICROP_DEPLOYED = True

_ONNX_NAME, _CONFIG_NAME = MODELS[MODEL_KEY]
ONNX_PATH = BASE_DIR / _ONNX_NAME
CONFIG_PATH = BASE_DIR / _CONFIG_NAME
RESULTS_PATH = BASE_DIR / f"results_face_eval.{MODEL_KEY}.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# DATA
# ============================================================


@dataclass
class Sample:
    path: Path
    label_index: int


def load_index(root: Path, limit_per_class: int | None) -> list[Sample]:
    """Collect labelled image paths from a class-per-directory layout."""

    samples: list[Sample] = []

    known = {label.lower(): index for index, label in enumerate(SHARED_LABELS)}

    # Common spellings that mean the same class, so a FER2013 export
    # does not need renaming before it can be evaluated.
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "sadness": "sad",
        "happiness": "happy",
        "joy": "happy",
        "surprised": "surprise",
        "disgusted": "disgust",
        "fearful": "fear",
        "calm": "neutral",
    }

    for directory in sorted(p for p in root.iterdir() if p.is_dir()):

        name = directory.name.strip().lower()
        name = aliases.get(name, name)

        if name not in known:
            print(f"  [skip] {directory.name!r} is not a shared label")
            continue

        paths = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )

        if limit_per_class is not None:
            paths = paths[:limit_per_class]

        samples.extend(Sample(path=p, label_index=known[name]) for p in paths)

    return samples


# ============================================================
# METRICS
# ============================================================


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-F1 over all seven classes, plus the per-class vector."""

    scores = np.zeros(NUM_SHARED_LABELS)

    for index in range(NUM_SHARED_LABELS):

        true_positive = int(np.sum((predictions == index) & (labels == index)))
        false_positive = int(np.sum((predictions == index) & (labels != index)))
        false_negative = int(np.sum((predictions != index) & (labels == index)))

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )

        scores[index] = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return float(scores.mean()), scores


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    """Weighted mean gap between confidence and accuracy across bins."""

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0

    for low, high in zip(edges[:-1], edges[1:]):

        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())

        if count == 0:
            continue

        error += (count / len(labels)) * abs(
            correct[in_bin].mean() - confidence[in_bin].mean()
        )

    return float(error)


# ============================================================
# ONE CONFIGURATION
# ============================================================


@dataclass
class Configuration:
    name: str
    crop_mode: CropMode
    use_tta: bool
    temperature: float
    description: str

    # Per-class logit offset, as stored in the model's config. The
    # server applies it on every frame, so a harness that leaves it at
    # zero is not measuring the deployed pipeline. Leaving it out is
    # what let v3 ship while predicting 'neutral' on 2 faces in 693:
    # the table showed a neutral F1 near zero and that was taken as the
    # model's ceiling rather than a correctable prior error.
    logit_bias: np.ndarray | None = None

    # Average logits over MULTICROP_RATIOS instead of scoring one crop.
    multicrop: bool = False


def run_configuration(
    configuration: Configuration,
    samples: list[Sample],
    detector,
    classifier: EmotionClassifier,
) -> dict:
    """Score one configuration over the whole sample set."""

    classifier.use_tta = configuration.use_tta
    classifier.temperature = configuration.temperature

    logits_out = []
    labels_out = []
    qualities = []

    detected = 0
    started = time.perf_counter()

    for sample in samples:

        image = cv2.imread(str(sample.path))

        if image is None:
            continue

        box = detector.detect(image)

        if box is None:
            # A miss is a failure of the configuration, not a sample to
            # drop. Excluding undetected faces would flatter whichever
            # detector finds fewest faces, which is backwards.
            labels_out.append(sample.label_index)
            logits_out.append(None)
            continue

        crop = crop_face(
            image, box, configuration.crop_mode, classifier.image_size
        )

        if crop is None or crop.size == 0:
            labels_out.append(sample.label_index)
            logits_out.append(None)
            continue

        detected += 1

        qualities.append(assess_quality(image, crop, box, None).overall)

        if configuration.multicrop:
            views = [
                crop_face(image, box, CropMode.ALIGNED,
                          classifier.image_size, eye_distance_ratio=ratio)
                for ratio in MULTICROP_RATIOS
            ]
            views = [v for v in views if v is not None and v.size]
            logits_out.append(
                classifier.logits_many(views) if views
                else classifier.logits(crop)
            )
        else:
            logits_out.append(classifier.logits(crop))

        labels_out.append(sample.label_index)

    elapsed = time.perf_counter() - started

    labels = np.array(labels_out)

    # Undetected faces get a uniform distribution: the configuration
    # genuinely had nothing to say, and a uniform vector scores as the
    # chance-level guess it is.
    uniform = np.full(NUM_SHARED_LABELS, 1.0 / NUM_SHARED_LABELS)

    bias = (
        np.zeros(NUM_SHARED_LABELS, dtype=np.float32)
        if configuration.logit_bias is None
        else np.asarray(configuration.logit_bias, dtype=np.float32)
    )

    probabilities = np.array([
        uniform if entry is None
        else softmax(entry / configuration.temperature + bias)
        for entry in logits_out
    ])

    predictions = probabilities.argmax(axis=1)

    accuracy = float((predictions == labels).mean())
    macro, per_class = macro_f1(labels, predictions)

    return {
        "name": configuration.name,
        "description": configuration.description,
        "crop_mode": configuration.crop_mode.value,
        "tta": configuration.use_tta,
        "temperature": configuration.temperature,
        "logit_bias": None if configuration.logit_bias is None
                      else [float(v) for v in bias],
        "multicrop": configuration.multicrop,
        "samples": len(labels),
        "detection_rate": detected / max(len(labels), 1),
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "accuracy": accuracy,
        "macro_f1": macro,
        "per_class_f1": {
            label: float(per_class[index])
            for index, label in enumerate(SHARED_LABELS)
        },
        "ece": expected_calibration_error(probabilities, labels),
        "ms_per_image": elapsed / max(len(labels), 1) * 1000.0,
        "_logits": logits_out,
        "_labels": labels,
    }


# ============================================================
# TEMPERATURE
# ============================================================


def fit_temperature(logits_list, labels: np.ndarray) -> float:
    """Golden-section search for the temperature minimising NLL."""

    usable = [
        (entry, label)
        for entry, label in zip(logits_list, labels)
        if entry is not None
    ]

    if not usable:
        return 1.0

    logits = np.array([entry for entry, _ in usable])
    targets = np.array([label for _, label in usable])

    def nll(temperature: float) -> float:
        probabilities = softmax(logits / temperature)
        chosen = probabilities[np.arange(len(targets)), targets]
        return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.05, 10.0
    c, d = b - golden * (b - a), a + golden * (b - a)
    fc, fd = nll(c), nll(d)

    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = nll(d)
        if abs(b - a) < 1e-4:
            break

    return float((a + b) / 2.0)


def fit_calibration(
    logits_list,
    labels: np.ndarray,
    seed: int = 0,
    holdout: float = 0.5,
) -> dict:
    """Fit temperature and a per-class logit bias, honestly reported.

    Temperature alone cannot fix a prior error. It divides every logit
    by the same scalar, so it never changes which class is largest -
    a model that systematically under-predicts one class still
    under-predicts it at every temperature. v3 shipped predicting
    'neutral' on 2 of 693 unseen faces; no temperature could have
    moved that, and the shipped one made it worse by sharpening.

    A per-class additive bias can, because it is not a monotonic
    rescaling of the whole vector. Fitted by coordinate descent on NLL,
    which is enough for seven parameters.

    The fit runs on half the data and the returned metrics are computed
    on the other half. Fitting seven free parameters and then quoting
    the score on the same rows is how a calibration that does nothing
    still looks like it helps.
    """

    usable = [
        (entry, label)
        for entry, label in zip(logits_list, labels)
        if entry is not None
    ]

    if len(usable) < 50:
        return {"temperature": 1.0, "logit_bias": [0.0] * NUM_SHARED_LABELS}

    logits = np.array([entry for entry, _ in usable])
    targets = np.array([label for _, label in usable])

    split = np.random.default_rng(seed).permutation(len(targets))

    if holdout <= 0.0:
        # Caller has a separate test set, so every row here can go into
        # the fit. The reported numbers are then in-sample and say so.
        fit_idx = report_idx = split
    else:
        cut = int(len(split) * (1.0 - holdout))
        fit_idx, report_idx = split[:cut], split[cut:]

    def nll(subset, temperature, bias) -> float:
        probabilities = softmax(logits[subset] / temperature + bias)
        chosen = probabilities[np.arange(len(subset)), targets[subset]]
        return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())

    temperature = fit_temperature(
        [logits[i] for i in fit_idx], targets[fit_idx]
    )

    bias = np.zeros(NUM_SHARED_LABELS, dtype=np.float64)
    grid = np.linspace(-3.0, 4.0, 141)

    previous = None
    for _ in range(40):
        for k in range(NUM_SHARED_LABELS):
            losses = []
            for value in grid:
                candidate = bias.copy()
                candidate[k] = value
                losses.append(nll(fit_idx, temperature, candidate))
            bias[k] = grid[int(np.argmin(losses))]

        current = nll(fit_idx, temperature, bias)
        if previous is not None and abs(previous - current) < 1e-6:
            break
        previous = current

    # Only differences matter - softmax is shift invariant - so centre
    # the vector. Keeps the stored numbers readable as "relative to the
    # average class" rather than drifting by an arbitrary constant.
    bias -= bias.mean()

    def score(subset, temperature_, bias_) -> dict:
        probabilities = softmax(logits[subset] / temperature_ + bias_)
        predictions = probabilities.argmax(axis=1)
        macro, per_class = macro_f1(targets[subset], predictions)
        return {
            "accuracy": float((predictions == targets[subset]).mean()),
            "macro_f1": macro,
            "neutral_f1": float(per_class[SHARED_LABELS.index("neutral")]),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
        }

    zero = np.zeros(NUM_SHARED_LABELS)

    return {
        "temperature": float(temperature),
        "logit_bias": [float(v) for v in bias],
        "fit_samples": int(len(fit_idx)),
        "report_samples": int(len(report_idx)),
        "heldout_uncalibrated": score(report_idx, 1.0, zero),
        "heldout_temperature_only": score(report_idx, temperature, zero),
        "heldout_calibrated": score(report_idx, temperature, bias),
    }


# ============================================================
# SMOOTHING (on a simulated stream)
# ============================================================


def evaluate_smoothing(result: dict, noise: float = 0.35, seed: int = 0) -> dict:
    """What temporal smoothing buys, simulated over a synthetic stream.

    A still-image test set cannot measure a temporal filter, so this
    constructs the situation the filter exists for: one held expression
    observed over many frames, where a fraction of the frames are
    degraded. Each sample becomes a short clip whose frames are its real
    logits plus noise, and `noise` is the per-frame logit standard
    deviation.

    This is a simulation and should be reported as one. It shows the
    filter behaves as designed under frame noise; it does not claim a
    real-world accuracy number.
    """

    rng = np.random.default_rng(seed)

    frames_per_clip = 12

    labels = result["_labels"]

    single_correct = 0
    smoothed_correct = 0
    counted = 0

    for entry, label in zip(result["_logits"], labels):

        if entry is None:
            continue

        counted += 1

        smoother = ProbabilitySmoother()

        last_single = 0

        for frame in range(frames_per_clip):

            noisy = entry + rng.normal(0.0, noise, size=entry.shape)

            probabilities = softmax(noisy)

            last_single = int(probabilities.argmax())

            # Quality varies frame to frame the way it does on a real
            # stream; the smoother is supposed to weight by it.
            quality = float(np.clip(rng.uniform(0.35, 1.0), 0.0, 1.0))

            smoother.update(probabilities, quality, now=frame * 0.1)

        single_correct += int(last_single == label)
        smoothed_correct += int(smoother.label_index == label)

    if counted == 0:
        return {"frames_per_clip": frames_per_clip, "noise": noise}

    return {
        "frames_per_clip": frames_per_clip,
        "noise_sigma": noise,
        "last_frame_accuracy": single_correct / counted,
        "smoothed_accuracy": smoothed_correct / counted,
        "clips": counted,
    }


# ============================================================
# REPORTING
# ============================================================


def print_table(results: list[dict]) -> None:

    print()
    print("=" * 96)
    print("FACE PIPELINE - INFERENCE CONFIGURATIONS")
    print("=" * 96)

    header = (
        f"{'Configuration':<26} {'Detect':>7} {'Acc':>8} {'MacroF1':>9} "
        f"{'ECE':>7} {'Quality':>8} {'ms/img':>8}"
    )
    print(header)
    print("-" * 96)

    baseline = results[0]

    for result in results:

        delta = (result["accuracy"] - baseline["accuracy"]) * 100

        marker = "" if result is baseline else f"  ({delta:+.2f})"

        print(
            f"{result['name']:<26} "
            f"{result['detection_rate']*100:>6.1f}% "
            f"{result['accuracy']*100:>7.2f}% "
            f"{result['macro_f1']*100:>8.2f}% "
            f"{result['ece']:>7.4f} "
            f"{result['mean_quality']*100:>7.1f}% "
            f"{result['ms_per_image']:>7.1f}" + marker
        )

    print("-" * 96)

    print("\nPer-class F1 (%)")
    print(f"{'Configuration':<26}" + "".join(f"{label:>10}" for label in SHARED_LABELS))

    for result in results:
        row = "".join(
            f"{result['per_class_f1'][label]*100:>10.1f}" for label in SHARED_LABELS
        )
        print(f"{result['name']:<26}{row}")

    print("=" * 96)


# ============================================================
# ROLL TEST
# ============================================================


def run_roll_test(
    samples: list[Sample],
    detector,
    classifier: EmotionClassifier,
    angles=(0, 5, 10, 15, 20, 25, 30),
    temperature: float = 1.0,
    logit_bias: np.ndarray | None = None,
) -> list[dict]:
    """Where does the aligned warp start earning its resampling cost?

    Rotating the whole frame rotates the face inside it, so the roll is
    known exactly rather than estimated. Both crop modes then see the
    identical rotated image, and the only difference between them is
    whether the warp undoes the rotation.

    Scored under the deployed calibration. A logit bias is not a
    monotonic rescaling, so it reorders predictions and the crossover
    angle moves with it - scoring at T=1 with no bias answers the
    question for a pipeline that is not the one being configured.
    """

    classifier.use_tta = False
    classifier.temperature = temperature
    classifier.logit_bias = (
        np.zeros(NUM_SHARED_LABELS, dtype=np.float32)
        if logit_bias is None
        else np.asarray(logit_bias, dtype=np.float32)
    )

    images = []
    labels = []

    for sample in samples:
        image = cv2.imread(str(sample.path))
        if image is not None:
            images.append(image)
            labels.append(sample.label_index)

    labels = np.array(labels)

    print(f"\n{'roll':>6} {'box crop':>10} {'aligned':>10} {'delta':>8}   detected")
    print("-" * 52)

    rows = []

    for angle in angles:

        box_crops, aligned_crops, kept = [], [], []

        for index, image in enumerate(images):

            if angle:
                height, width = image.shape[:2]
                matrix = cv2.getRotationMatrix2D(
                    (width / 2, height / 2), float(angle), 1.0
                )
                rotated = cv2.warpAffine(
                    image, matrix, (width, height),
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                rotated = image

            box = detector.detect(rotated)

            if box is None or box.landmarks is None:
                continue

            first = crop_face(rotated, box, CropMode.LEGACY,
                              classifier.image_size)
            second = crop_face(rotated, box, CropMode.ALIGNED,
                               classifier.image_size)

            if first is None or second is None:
                continue

            box_crops.append(first)
            aligned_crops.append(second)
            kept.append(index)

        if not kept:
            print(f"{angle:>5}deg   no faces detected")
            continue

        truth = labels[kept]

        def accuracy(crops):
            predictions = [
                int(np.argmax(classifier.predict(crop))) for crop in crops
            ]
            return float(np.mean(np.array(predictions) == truth)) * 100

        box_accuracy = accuracy(box_crops)
        aligned_accuracy = accuracy(aligned_crops)

        rows.append({
            "roll_degrees": angle,
            "box_crop": box_accuracy,
            "aligned": aligned_accuracy,
            "delta": aligned_accuracy - box_accuracy,
            "detected": len(kept),
        })

        print(f"{angle:>5}deg {box_accuracy:>9.2f}% {aligned_accuracy:>9.2f}% "
              f"{aligned_accuracy - box_accuracy:>+7.2f}   {len(kept)}/{len(images)}",
              flush=True)

    print("-" * 52)

    crossover = next((row["roll_degrees"] for row in rows if row["delta"] > 0), None)

    if crossover is None:
        print("Aligned never beats the plain box crop at any roll tested.")
        print("Consider CropMode.LEGACY as the deployed default.")
    else:
        print(f"Aligned starts paying off at about {crossover} degrees.")
        print(f"ROLL_ALIGN_THRESHOLD_DEGREES is currently "
              f"{ROLL_ALIGN_THRESHOLD_DEGREES:.0f}.")

    return rows


# ============================================================
# ENTRY POINT
# ============================================================


def build_classifier(prefer_gpu: bool = True) -> tuple[EmotionClassifier, dict]:

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    assert_matches_config(
        {int(key): value for key, value in config["id2label"].items()}
    )

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=select_providers(prefer_gpu)
    )

    print(f"ONNX provider: {describe_providers(session)}")

    classifier = EmotionClassifier(
        session=session,
        input_name=session.get_inputs()[0].name,
        image_size=int(config.get("image_size", 224)),
        mean=np.array(config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32),
        std=np.array(config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32),
        temperature=1.0,
        use_tta=False,
    )

    return classifier, config


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with one subdirectory per shared emotion label.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Cap images per class, for a quick run.",
    )
    parser.add_argument(
        "--fit-temperature",
        action="store_true",
        help="Fit the calibration temperature and write it to the model's config.",
    )
    parser.add_argument(
        "--calibration-holdout",
        type=float,
        default=0.5,
        help=(
            "Fraction of --data held back from the calibration fit and "
            "used for the reported numbers. Set to 0 when --data is "
            "already a dedicated calibration split and the real test set "
            "is scored separately; the printed table is then in-sample."
        ),
    )
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help=(
            "Fit temperature AND the per-class logit bias, and write both "
            "to the model's config. Prefer this over --fit-temperature: "
            "temperature cannot correct a prior error because it never "
            "reorders predictions. Fitted on half the data and reported "
            "on the other half."
        ),
    )
    parser.add_argument(
        "--sweep-crop",
        action="store_true",
        help="Sweep the inter-ocular crop ratio instead of comparing configurations.",
    )
    parser.add_argument(
        "--roll-test",
        action="store_true",
        help="Apply a known head roll and report where aligned beats the box crop.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when a GPU provider is available.",
    )

    arguments = parser.parse_args()

    if not arguments.data.is_dir():
        raise SystemExit(f"{arguments.data} is not a directory.")

    print(f"Indexing {arguments.data} ...")

    samples = load_index(arguments.data, arguments.limit_per_class)

    if not samples:
        raise SystemExit(
            "No labelled images found. Expected one subdirectory per emotion, "
            f"named from: {', '.join(SHARED_LABELS)}"
        )

    counts = np.bincount(
        [s.label_index for s in samples], minlength=NUM_SHARED_LABELS
    )

    print(f"{len(samples)} images")
    for index, label in enumerate(SHARED_LABELS):
        print(f"  {label:<9} {counts[index]:>6}")

    classifier, config = build_classifier(prefer_gpu=not arguments.cpu)
    detector = build_detector()

    print(f"\nDetector: {detector.name}")

    if detector.name == "haar":
        print(
            "  WARNING: no landmarks available, so the aligned rows below use "
            "geometric pseudo-landmarks. Run setup_models.py for the real thing."
        )

    # ----------------------------------------------------
    # CROP SWEEP
    # ----------------------------------------------------

    if arguments.sweep_crop:

        import face_pipeline

        original = face_pipeline.EYE_DISTANCE_RATIO

        print("\nSweeping the inter-ocular crop ratio")
        # Swept under the deployed calibration, not at T=1 with no bias.
        # The optimum genuinely moves: uncalibrated, the model barely
        # emits 'neutral' at any framing, so the sweep sees nothing to
        # gain from a wider crop and settles around 0.42. With the
        # stored bias applied, a wider crop is what makes neutral
        # reachable, and the optimum moves well below that. Sweeping a
        # configuration nobody serves is how the original 0.38 was
        # chosen.
        swept_temperature = float(config.get("temperature", 1.0))
        swept_bias = config.get("logit_bias")
        swept_bias = (
            None if swept_bias is None
            else np.asarray(swept_bias, dtype=np.float32)
        )

        print(f"Calibration: T={swept_temperature:.3f}, "
              f"bias={'stored' if swept_bias is not None else 'none'}")
        print(f"{'ratio':>7} {'accuracy':>10} {'macro-F1':>10} {'neutral-F1':>11}")
        print("-" * 42)

        for ratio in (0.18, 0.21, 0.24, 0.27, 0.30, 0.34,
                      0.38, 0.42, 0.46, 0.50):

            face_pipeline.EYE_DISTANCE_RATIO = ratio

            result = run_configuration(
                Configuration(
                    f"ratio {ratio}", CropMode.ALIGNED, False,
                    swept_temperature, "", logit_bias=swept_bias,
                ),
                samples, detector, classifier,
            )

            print(
                f"{ratio:>7.2f} {result['accuracy']*100:>9.2f}% "
                f"{result['macro_f1']*100:>9.2f}% "
                f"{result['per_class_f1']['neutral']*100:>10.2f}%"
            )

        face_pipeline.EYE_DISTANCE_RATIO = original
        return

    # ----------------------------------------------------
    # ROLL TEST
    # ----------------------------------------------------

    if arguments.roll_test:

        rows = run_roll_test(
            samples, detector, classifier,
            temperature=float(config.get("temperature", 1.0)),
            logit_bias=config.get("logit_bias"),
        )

        with open(BASE_DIR / "results_roll_test.json", "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

        print(f"\nWrote {BASE_DIR / 'results_roll_test.json'}")
        return

    # ----------------------------------------------------
    # CONFIGURATIONS
    # ----------------------------------------------------

    configurations = [
        Configuration(
            f"1. box + {LEGACY_MARGIN:.0%}", CropMode.LEGACY, False, 1.0,
            "Plain detector box plus margin, no alignment.",
        ),
        Configuration(
            "2. + aligned crop", CropMode.ALIGNED, False, 1.0,
            f"Eye-line similarity transform, IOD ratio {EYE_DISTANCE_RATIO}.",
        ),
        Configuration(
            "3. adaptive crop", CROP_MODE_DEPLOYED, False, 1.0,
            f"Box crop below {ROLL_ALIGN_THRESHOLD_DEGREES:.0f} deg roll, "
            f"aligned warp above it. The deployed default.",
        ),
        Configuration(
            "4. + flip TTA", CropMode.ADAPTIVE, True, 1.0,
            "Mean of the logits of the crop and its mirror.",
        ),
        Configuration(
            "5. multi-crop", CROP_MODE_DEPLOYED, False, 1.0,
            f"Mean logits over aligned crops at ratios "
            f"{', '.join(str(r) for r in MULTICROP_RATIOS)}. The deployed "
            f"geometry.",
            multicrop=True,
        ),
    ]

    results = []

    for configuration in configurations:
        print(f"\nRunning {configuration.name} ...", flush=True)
        results.append(
            run_configuration(configuration, samples, detector, classifier)
        )

    # ----------------------------------------------------
    # CALIBRATION
    # ----------------------------------------------------

    best = results[-1]

    temperature = fit_temperature(best["_logits"], best["_labels"])

    calibrated = run_configuration(
        Configuration(
            "6. + calibration", CROP_MODE_DEPLOYED, USE_TTA_DEPLOYED, temperature,
            f"Temperature scaling, T={temperature:.3f}.",
        ),
        samples, detector, classifier,
    )

    results.append(calibrated)

    # The row that corresponds to what server.py actually serves:
    # stored temperature *and* stored logit_bias, not a freshly fitted
    # temperature alone. Without it the table's best row is still an
    # un-deployed configuration, which is how the shipped pipeline came
    # to be worse than every line of its own results file.
    stored_bias = config.get("logit_bias")

    if stored_bias is not None:
        deployed = run_configuration(
            Configuration(
                "7. deployed (multi-crop + T + bias)", CROP_MODE_DEPLOYED, USE_TTA_DEPLOYED,
                float(config.get("temperature", 1.0)),
                f"What server.py serves: multi-crop, "
                f"T={float(config.get('temperature', 1.0)):.3f}, "
                f"with the stored per-class logit bias.",
                logit_bias=np.asarray(stored_bias, dtype=np.float32),
                multicrop=USE_MULTICROP_DEPLOYED,
            ),
            samples, detector, classifier,
        )
        results.append(deployed)

    print_table(results)

    # ----------------------------------------------------
    # SMOOTHING
    # ----------------------------------------------------

    smoothing = evaluate_smoothing(calibrated, seed=arguments.seed)

    if "smoothed_accuracy" in smoothing:
        print("\nTemporal smoothing (SIMULATED stream, not a real-world number)")
        print(f"  {smoothing['clips']} clips x {smoothing['frames_per_clip']} frames, "
              f"logit noise sigma {smoothing['noise_sigma']}")
        print(f"  last frame only  {smoothing['last_frame_accuracy']*100:.2f}%")
        print(f"  smoothed         {smoothing['smoothed_accuracy']*100:.2f}%")

    # ----------------------------------------------------
    # PERSIST
    # ----------------------------------------------------

    # Calibrate on the logits the server will actually produce: the
    # deployed crop mode at the deployed TTA setting. `best` above is
    # the TTA row, which is a different distribution of logits - fitting
    # on it and serving without it is the same category of error as
    # fitting the temperature on FER2013 and serving on a webcam.
    deployed_match = [
        row for row in results
        if row["crop_mode"] == CROP_MODE_DEPLOYED.value
        and row["tta"] == USE_TTA_DEPLOYED
        and row.get("multicrop") == USE_MULTICROP_DEPLOYED
        and row.get("logit_bias") is None
    ]

    calibration_source = deployed_match[0] if deployed_match else best

    # Held before the pop below, which strips the arrays so the JSON
    # payload stays a results file rather than a logit dump.
    calibration_logits = calibration_source["_logits"]
    calibration_labels = calibration_source["_labels"]

    for result in results:
        result.pop("_logits", None)
        result.pop("_labels", None)

    payload = {
        "data": str(arguments.data),
        "detector": detector.name,
        "samples": len(samples),
        "fitted_temperature": temperature,
        "configurations": results,
        "smoothing_simulation": smoothing,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {RESULTS_PATH}")

    if arguments.fit_calibration:
        calibration = fit_calibration(
            calibration_logits, calibration_labels, seed=arguments.seed,
            holdout=arguments.calibration_holdout,
        )

        print("\nCalibration (fitted on half, reported on the other half)")
        print(f"  temperature {calibration['temperature']:.4f}")
        print("  logit_bias  " + "  ".join(
            f"{l}={v:+.2f}"
            for l, v in zip(SHARED_LABELS, calibration["logit_bias"])
        ))
        print(f"\n  {'stage':<22} {'acc':>7} {'macroF1':>9} "
              f"{'neutralF1':>10} {'meanconf':>9}")
        for key, title in (
            ("heldout_uncalibrated", "uncalibrated"),
            ("heldout_temperature_only", "temperature only"),
            ("heldout_calibrated", "temperature + bias"),
        ):
            row = calibration[key]
            print(f"  {title:<22} {row['accuracy']*100:6.2f}% "
                  f"{row['macro_f1']*100:8.2f}% {row['neutral_f1']*100:9.2f}% "
                  f"{row['mean_confidence']*100:8.1f}%")

        config["temperature"] = calibration["temperature"]
        config["logit_bias"] = calibration["logit_bias"]
        config["_calibration_fit"] = {
            "data": str(arguments.data),
            "fit_samples": calibration["fit_samples"],
            "report_samples": calibration["report_samples"],
            "heldout_uncalibrated": calibration["heldout_uncalibrated"],
            "heldout_temperature_only": calibration["heldout_temperature_only"],
            "heldout_calibrated": calibration["heldout_calibrated"],
        }

        with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)

        print(f"\nWrote temperature and logit_bias to {CONFIG_PATH}")

    elif arguments.fit_temperature:
        config["temperature"] = temperature
        with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)
        print(f"Wrote temperature {temperature:.4f} to {CONFIG_PATH}")
    else:
        print(
            f"Fitted temperature is {temperature:.4f}. "
            f"Re-run with --fit-calibration to store temperature and "
            f"logit_bias in {CONFIG_PATH.name}."
        )


if __name__ == "__main__":
    main()
