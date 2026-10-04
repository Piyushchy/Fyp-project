"""
Tests for the deployment side: shared labels, face pipeline, fusion.

    python test_deployment.py
    pytest test_deployment.py -v

Tests needing the ONNX model or the YuNet asset skip themselves when
those files are absent, so the suite is useful on a fresh clone.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

import face_pipeline as fp
from fusion import (
    EPSILON,
    AudioEvidence,
    FusionConfig,
    FusionEngine,
    TextEvidence,
    agreement,
    certainty,
    informativeness,
    recency,
)
from labels import (
    NUM_SHARED_LABELS,
    SHARED_LABEL2ID,
    SHARED_LABELS,
    assert_matches_config,
    to_mapping,
    to_vector,
)

BASE_DIR = Path(__file__).resolve().parent


class Skip(Exception):
    """Raised to mark a test as skipped rather than failed."""


# ============================================================
# HELPERS
# ============================================================


def peaked(label: str, probability: float) -> np.ndarray:
    """A distribution with exactly `probability` on `label`."""

    vector = np.full(NUM_SHARED_LABELS, (1.0 - probability) / (NUM_SHARED_LABELS - 1))
    vector[SHARED_LABEL2ID[label]] = probability

    return vector


UNIFORM = np.full(NUM_SHARED_LABELS, 1.0 / NUM_SHARED_LABELS)


def evidence(vector: np.ndarray, tokens: int = 12, timestamp: float = 0.0):
    return TextEvidence(
        probabilities=vector,
        token_count=tokens,
        text="message",
        timestamp=timestamp,
        label=SHARED_LABELS[int(vector.argmax())],
        confidence=float(vector.max()),
    )


def synthetic_face_frame(roll_degrees: float = 0.0) -> tuple[np.ndarray, fp.FaceBox]:
    """A plain frame plus a hand-specified box, for geometry tests."""

    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.randn(frame, (130, 130, 130), (35, 35, 35))

    radians = math.radians(roll_degrees)

    centre = np.array([300.0, 220.0])
    half = 40.0

    offset = np.array([math.cos(radians), math.sin(radians)]) * half

    eye_left = centre - offset
    eye_right = centre + offset

    landmarks = np.array(
        [
            eye_left,
            eye_right,
            [300.0, 260.0],
            [270.0, 300.0],
            [330.0, 300.0],
        ],
        dtype=np.float32,
    )

    box = fp.FaceBox(200, 140, 200, 200, 0.95, landmarks)

    return frame, box


# ============================================================
# SHARED LABELS
# ============================================================


def test_shared_space_matches_the_exported_model():
    """A mismatch here would make fusion compare different emotions."""

    config = json.loads(
        (BASE_DIR / "emotion_config.json").read_text(encoding="utf-8")
    )

    id2label = {int(key): value for key, value in config["id2label"].items()}

    assert_matches_config(id2label)

    try:
        assert_matches_config({0: "happy", 1: "sad"})
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong label order should raise")


def test_vector_and_mapping_round_trip():

    vector = peaked("happy", 0.6)

    assert to_vector(to_mapping(vector)) == [float(v) for v in vector]

    # An absent label reads as zero rather than raising.
    assert to_vector({"happy": 1.0})[SHARED_LABEL2ID["sad"]] == 0.0


# ============================================================
# ALIGNMENT GEOMETRY
# ============================================================


def test_alignment_puts_the_eyes_at_canonical_positions():

    matrix = fp.similarity_transform(
        np.float32([100, 100]), np.float32([160, 120]), 224
    )

    mapped = cv2.transform(np.float32([[[100, 100], [160, 120]]]), matrix)[0]

    expected_y = fp.EYE_Y_RATIO * 224
    expected_left = (0.5 - fp.EYE_DISTANCE_RATIO / 2) * 224
    expected_right = (0.5 + fp.EYE_DISTANCE_RATIO / 2) * 224

    assert abs(mapped[0][0] - expected_left) < 1e-3
    assert abs(mapped[1][0] - expected_right) < 1e-3
    assert abs(mapped[0][1] - expected_y) < 1e-3
    assert abs(mapped[1][1] - expected_y) < 1e-3


def test_alignment_removes_head_roll():
    """The whole point of the eye-line transform."""

    for roll in (-40, -15, 0, 15, 40):

        radians = math.radians(roll)
        offset = np.array([math.cos(radians), math.sin(radians)]) * 50.0

        eye_left = np.float32([300, 220] - offset)
        eye_right = np.float32([300, 220] + offset)

        matrix = fp.similarity_transform(eye_left, eye_right, 224)

        mapped = cv2.transform(
            np.float32([[eye_left, eye_right]]), matrix
        )[0]

        assert abs(mapped[1][1] - mapped[0][1]) < 1e-3, f"roll {roll} not corrected"


def test_alignment_is_scale_invariant():
    """A face twice as far away must produce the same canonical crop."""

    near = fp.similarity_transform(np.float32([200, 200]), np.float32([300, 200]), 224)
    far = fp.similarity_transform(np.float32([250, 200]), np.float32([300, 200]), 224)

    near_pts = cv2.transform(np.float32([[[200, 200], [300, 200]]]), near)[0]
    far_pts = cv2.transform(np.float32([[[250, 200], [300, 200]]]), far)[0]

    assert np.allclose(near_pts, far_pts, atol=1e-3)


def test_degenerate_eye_line_is_rejected():

    try:
        fp.similarity_transform(np.float32([100, 100]), np.float32([100, 100]), 224)
    except ValueError:
        pass
    else:
        raise AssertionError("coincident eye points should raise")


def test_crop_modes_produce_the_right_shape_and_differ():

    frame, box = synthetic_face_frame()

    aligned = fp.crop_face(frame, box, fp.CropMode.ALIGNED, 224)
    legacy = fp.crop_face(frame, box, fp.CropMode.LEGACY, 224)

    assert aligned.shape == (224, 224, 3)
    assert legacy.shape == (224, 224, 3)

    # If these were identical the comparison in eval_face.py would be
    # measuring nothing.
    assert not np.array_equal(aligned, legacy)


def test_missing_landmarks_fall_back_without_crashing():

    frame, box = synthetic_face_frame()

    bare = fp.FaceBox(box.x, box.y, box.w, box.h, 0.55, None)

    crop = fp.crop_face(frame, bare, fp.CropMode.ALIGNED, 224)

    assert crop is not None and crop.shape == (224, 224, 3)

    # Without landmarks there is no pose information, so the score must
    # be middling rather than confident.
    assert 0.0 < fp._pose_score(bare) < 1.0


# ============================================================
# QUALITY
# ============================================================


def test_quality_factors_respond_to_their_own_degradation():

    frame, box = synthetic_face_frame()
    crop = fp.crop_face(frame, box, fp.CropMode.ALIGNED, 224)

    baseline = fp.assess_quality(frame, crop, box, None)

    assert 0.0 < baseline.overall <= 1.0

    # --- blur ---
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    blurred_crop = fp.crop_face(blurred_frame, box, fp.CropMode.ALIGNED, 224)
    blurred = fp.assess_quality(blurred_frame, blurred_crop, box, None)

    assert blurred.sharpness < baseline.sharpness
    assert blurred.overall < baseline.overall

    # --- exposure ---
    dark_frame = np.zeros_like(frame)
    cv2.randn(dark_frame, (20, 20, 20), (8, 8, 8))
    dark_crop = fp.crop_face(dark_frame, box, fp.CropMode.ALIGNED, 224)
    dark = fp.assess_quality(dark_frame, dark_crop, box, None)

    assert dark.exposure < baseline.exposure

    # --- pose: nose pushed toward one eye ---
    yawed = fp.FaceBox(
        box.x, box.y, box.w, box.h, box.score,
        np.float32([[260, 220], [340, 220], [338, 260], [270, 300], [330, 300]]),
    )
    yawed_quality = fp.assess_quality(
        frame, fp.crop_face(frame, yawed, fp.CropMode.ALIGNED, 224), yawed, None
    )

    assert yawed_quality.pose < baseline.pose

    # --- size ---
    small = fp.FaceBox(300, 220, 50, 50, 0.95, box.landmarks)
    small_quality = fp.assess_quality(frame, crop, small, None)

    assert small_quality.size < baseline.size


def test_absent_quality_is_exactly_zero():
    """No face must contribute no weight at all, not a small one."""

    absent = fp.Quality.absent()

    assert absent.overall == 0.0

    engine = FusionEngine()

    result = engine.fuse(None, absent.overall, None, now=0.0)

    assert result.mode == "none" and result.label is None


def test_quality_names_its_weakest_factor():

    frame, box = synthetic_face_frame()

    dark = np.zeros_like(frame)
    cv2.randn(dark, (18, 18, 18), (6, 6, 6))

    quality = fp.assess_quality(
        dark, fp.crop_face(dark, box, fp.CropMode.ALIGNED, 224), box, None
    )

    assert quality.weakest() in quality.as_dict()
    assert quality.as_dict()["weakest"] == quality.weakest()


def test_iou_tracks_box_stability():

    box = fp.FaceBox(100, 100, 100, 100, 0.9, None)

    assert box.iou(None) == 1.0
    assert abs(box.iou(fp.FaceBox(100, 100, 100, 100, 0.9, None)) - 1.0) < 1e-9
    assert box.iou(fp.FaceBox(400, 400, 100, 100, 0.9, None)) == 0.0

    partial = box.iou(fp.FaceBox(150, 100, 100, 100, 0.9, None))
    assert 0.0 < partial < 1.0


# ============================================================
# SMOOTHING
# ============================================================


def test_smoother_resists_a_single_low_quality_contradiction():

    smoother = fp.ProbabilitySmoother()

    happy = peaked("happy", 0.95)
    sad = peaked("sad", 0.95)

    smoother.update(happy, 1.0, now=0.0)
    assert smoother.label == "happy"

    smoother.update(sad, 0.05, now=0.1)

    assert smoother.label == "happy", "one bad frame flipped the label"


def test_smoother_follows_a_sustained_change():

    smoother = fp.ProbabilitySmoother()

    smoother.update(peaked("happy", 0.95), 1.0, now=0.0)

    sad = peaked("sad", 0.95)

    for step in range(1, 40):
        smoother.update(sad, 1.0, now=step * 0.1)

    assert smoother.label == "sad"


def test_smoother_step_scales_with_quality():
    """A clean frame must move the state further than a degraded one."""

    target = peaked("sad", 0.95)

    moved = {}

    for quality in (1.0, 0.25):
        smoother = fp.ProbabilitySmoother()
        smoother.update(peaked("happy", 0.95), 1.0, now=0.0)
        smoother.update(target, quality, now=0.3)
        moved[quality] = smoother.state[SHARED_LABEL2ID["sad"]]

    assert moved[1.0] > moved[0.25]


def test_smoother_step_scales_with_elapsed_time():
    """Frame pacing must not change the effective smoothing window."""

    target = peaked("sad", 0.95)

    moved = {}

    for gap in (0.1, 1.0):
        smoother = fp.ProbabilitySmoother()
        smoother.update(peaked("happy", 0.95), 1.0, now=0.0)
        smoother.update(target, 1.0, now=gap)
        moved[gap] = smoother.state[SHARED_LABEL2ID["sad"]]

    assert moved[1.0] > moved[0.1]


def test_smoother_restarts_after_a_long_gap():

    smoother = fp.ProbabilitySmoother()

    smoother.update(peaked("happy", 0.95), 1.0, now=0.0)

    # Longer than MAX_SMOOTHING_GAP: the old state is no longer
    # evidence about the present.
    smoother.update(peaked("sad", 0.95), 1.0, now=fp.MAX_SMOOTHING_GAP + 1.0)

    assert smoother.label == "sad"


def test_smoothed_state_stays_a_distribution():

    rng = np.random.default_rng(0)

    smoother = fp.ProbabilitySmoother()

    for step in range(60):
        vector = rng.random(NUM_SHARED_LABELS)
        vector /= vector.sum()
        state = smoother.update(vector, float(rng.random()), now=step * 0.1)

        assert abs(state.sum() - 1.0) < 1e-9
        assert np.all(state >= 0.0)
        assert np.all(np.isfinite(state))


# ============================================================
# FUSION FACTORS
# ============================================================


def test_certainty_spans_the_full_range():

    assert certainty(UNIFORM) < 1e-9

    assert certainty(peaked("happy", 0.999)) > 0.95

    # Monotone in peakedness.
    assert certainty(peaked("happy", 0.9)) > certainty(peaked("happy", 0.5))


def test_recency_halves_at_the_half_life_and_zeroes_at_the_cutoff():

    assert abs(recency(0.0, 30.0, 180.0) - 1.0) < 1e-9
    assert abs(recency(30.0, 30.0, 180.0) - 0.5) < 1e-9
    assert abs(recency(60.0, 30.0, 180.0) - 0.25) < 1e-9

    assert recency(180.0, 30.0, 180.0) == 0.0
    assert recency(1e6, 30.0, 180.0) == 0.0

    # Negative ages (clock skew) must not amplify anything.
    assert recency(-5.0, 30.0, 180.0) == 1.0


def test_informativeness_ramps_then_saturates():

    assert informativeness(0, 8.0) == 0.0
    assert abs(informativeness(4, 8.0) - 0.5) < 1e-9
    assert informativeness(8, 8.0) == 1.0
    assert informativeness(400, 8.0) == 1.0


def test_agreement_is_bounded_and_symmetric():

    a = peaked("happy", 0.8)
    b = peaked("angry", 0.8)

    assert abs(agreement(a, a) - 1.0) < 1e-9
    assert abs(agreement(a, b) - agreement(b, a)) < 1e-12

    for first, second in [(a, b), (a, UNIFORM), (UNIFORM, UNIFORM)]:
        assert 0.0 <= agreement(first, second) <= 1.0

    # Disjoint supports really do reach zero.
    left = np.zeros(NUM_SHARED_LABELS)
    left[0] = 1.0
    right = np.zeros(NUM_SHARED_LABELS)
    right[1] = 1.0
    assert agreement(left, right) < 1e-9

    assert agreement(np.zeros(NUM_SHARED_LABELS), a) == 0.0


# ============================================================
# FUSION POOL
# ============================================================


def test_flat_reliability_passes_a_single_modality_through_unchanged():
    """With nothing to discount, the pool must be the identity.

    Both the base weight and the per-class reliability are neutralised
    here, because this test is about the pool arithmetic rather than
    about the tuned configuration. With exponent exactly 1 and flat
    reliability the geometric mean of one distribution is that
    distribution, and any deviation is a bug in `_log_pool`.
    """

    config = FusionConfig.load()
    config.face.reliability = np.ones(NUM_SHARED_LABELS)
    config.text.reliability = np.ones(NUM_SHARED_LABELS)
    config.face.base_weight = 1.0
    config.text.base_weight = 1.0

    engine = FusionEngine(config)

    for label in SHARED_LABELS:
        result = engine.fuse(peaked(label, 0.7), 1.0, None, now=0.0)

        assert result.label == label
        assert abs(result.confidence - 0.7) < 1e-9


def test_base_weight_below_one_softens_a_single_modality():
    """A modality trusted at 0.8 observations must not report full confidence.

    This is the flip side of the identity test above: the tuned
    configuration discounts the face modality because it measures
    materially weaker cross-dataset, and that discount has to show up
    as lower confidence rather than being silently ignored.
    """

    config = FusionConfig.load()
    config.face.reliability = np.ones(NUM_SHARED_LABELS)
    config.face.base_weight = 0.8

    engine = FusionEngine(config)

    result = engine.fuse(peaked("happy", 0.7), 1.0, None, now=0.0)

    assert result.label == "happy"
    assert result.confidence < 0.7

    # Still a valid distribution, and still the same decision.
    assert abs(result.probabilities.sum() - 1.0) < 1e-9


def test_per_class_reliability_tilts_toward_the_stronger_class():

    engine = FusionEngine()

    order = sorted(
        SHARED_LABELS,
        key=lambda label: engine.config.face.reliability[SHARED_LABEL2ID[label]],
    )

    weakest, strongest = order[0], order[-1]

    weak = engine.fuse(peaked(weakest, 0.7), 1.0, None, now=0.0)
    strong = engine.fuse(peaked(strongest, 0.7), 1.0, None, now=0.0)

    assert strong.confidence > 0.7 > weak.confidence


def test_agreeing_modalities_sharpen():
    """A product of experts must beat either expert alone."""

    engine = FusionEngine()

    face = peaked("happy", 0.55)
    text = peaked("happy", 0.60)

    result = engine.fuse(face, 1.0, evidence(text), now=0.0)

    assert result.label == "happy"
    assert result.confidence > max(face.max(), text.max())
    assert result.mode == "fused"


def test_a_uniform_modality_contributes_nothing():
    """log(uniform) is constant and must cancel in the softmax."""

    engine = FusionEngine()

    face = peaked("sad", 0.7)

    alone = engine.fuse(face, 1.0, None, now=0.0)
    with_noise = engine.fuse(face, 1.0, evidence(UNIFORM), now=0.0)

    assert abs(alone.confidence - with_noise.confidence) < 1e-9
    assert with_noise.text.influence < 1e-9


def test_pool_degrades_to_each_single_modality():

    engine = FusionEngine()

    assert engine.fuse(peaked("sad", 0.7), 0.9, None, now=0.0).mode == "face_only"
    assert engine.fuse(None, 0.0, evidence(peaked("sad", 0.7)), now=0.0).mode == "text_only"

    nothing = engine.fuse(None, 0.0, None, now=0.0)
    assert nothing.mode == "none"
    assert nothing.label is None
    assert nothing.probabilities is None


def test_decayed_text_collapses_to_face_only():

    engine = FusionEngine()

    face = peaked("neutral", 0.5)
    text = evidence(peaked("angry", 0.85), timestamp=0.0)

    fresh = engine.fuse(face, 0.9, text, now=0.0)
    stale = engine.fuse(face, 0.9, text, now=1e6)

    assert fresh.label == "angry"
    assert stale.mode == "face_only"
    assert stale.label == "neutral"
    assert stale.text.influence == 0.0


def test_influences_always_sum_to_one_and_probs_are_valid():

    engine = FusionEngine()

    cases = [
        (peaked("happy", 0.5), 0.9, evidence(peaked("sad", 0.5))),
        (peaked("happy", 0.5), 0.9, None),
        (None, 0.0, evidence(peaked("sad", 0.5))),
        (UNIFORM, 0.9, evidence(UNIFORM)),
        (peaked("fear", 0.99), 0.02, evidence(peaked("fear", 0.99), tokens=1)),
    ]

    for face, quality, text in cases:

        result = engine.fuse(face, quality, text, now=0.0)

        assert abs(result.face.influence + result.text.influence - 1.0) < 1e-9

        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert np.all(np.isfinite(result.probabilities))
        assert np.all(result.probabilities >= 0.0)


def test_zeroed_class_does_not_produce_nan():
    """A hard zero in an input must not poison the log pool."""

    engine = FusionEngine()

    face = np.zeros(NUM_SHARED_LABELS)
    face[SHARED_LABEL2ID["happy"]] = 1.0

    text = np.zeros(NUM_SHARED_LABELS)
    text[SHARED_LABEL2ID["angry"]] = 1.0

    result = engine.fuse(face, 1.0, evidence(text), now=0.0)

    assert np.all(np.isfinite(result.probabilities))
    assert abs(result.probabilities.sum() - 1.0) < 1e-9


def test_conflict_needs_both_a_different_label_and_low_overlap():

    engine = FusionEngine()

    # Confidently opposed: a conflict.
    opposed = engine.fuse(
        peaked("happy", 0.8), 1.0, evidence(peaked("angry", 0.8)), now=0.0
    )
    assert opposed.conflicted

    # Same label, different confidence: not a conflict.
    same = engine.fuse(
        peaked("happy", 0.5), 1.0, evidence(peaked("happy", 0.9)), now=0.0
    )
    assert not same.conflicted

    # Two shrugs: different argmax but almost total overlap. Not a
    # conflict - neither modality is claiming anything.
    first = UNIFORM.copy()
    first[0] += 1e-4
    second = UNIFORM.copy()
    second[1] += 1e-4

    shrug = engine.fuse(first, 1.0, evidence(second), now=0.0)
    assert not shrug.conflicted

    # A single modality has nothing to disagree with.
    alone = engine.fuse(peaked("happy", 0.9), 1.0, None, now=0.0)
    assert not alone.conflicted
    assert alone.agreement == 1.0


def test_face_weight_tracks_quality_monotonically():

    engine = FusionEngine()

    text = evidence(peaked("sad", 0.7))

    previous = None

    for quality in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):

        result = engine.fuse(peaked("happy", 0.7), quality, text, now=0.0)

        if previous is not None:
            assert result.face.influence < previous

        previous = result.face.influence


def test_short_messages_carry_less_weight():

    engine = FusionEngine()

    face = peaked("neutral", 0.6)

    previous = 0.0

    for tokens in (1, 2, 4, 8):

        result = engine.fuse(
            face, 0.9, evidence(peaked("angry", 0.9), tokens=tokens), now=0.0
        )

        assert result.text.influence > previous
        previous = result.text.influence


def test_reliability_is_normalised_to_mean_one():
    """Only the ratios between classes should matter."""

    config = FusionConfig.load()

    for profile in (config.face, config.text):
        assert abs(profile.reliability.mean() - 1.0) < 1e-9
        assert len(profile.reliability) == NUM_SHARED_LABELS


def test_reliability_json_covers_every_shared_label():

    raw = json.loads((BASE_DIR / "reliability.json").read_text(encoding="utf-8"))

    for modality in ("face", "text"):
        per_class = raw[modality]["per_class_f1"]
        assert set(per_class) == set(SHARED_LABELS), (
            f"{modality} is missing: {set(SHARED_LABELS) - set(per_class)}"
        )


# ============================================================
# DETECTOR / MODEL (skipped when assets are absent)
# ============================================================


def test_detector_builds_and_reports_its_capabilities():

    detector = fp.build_detector()

    assert detector.name in ("yunet", "haar")

    # Only the landmark detector may claim landmarks.
    assert detector.provides_landmarks == (detector.name == "yunet")


def test_onnx_model_emits_seven_calibrated_probabilities():

    if not (BASE_DIR / "vit-emotion-v2.onnx").exists():
        raise Skip("ONNX model not present")

    import onnxruntime as ort

    config = json.loads(
        (BASE_DIR / "emotion_config.json").read_text(encoding="utf-8")
    )

    session = ort.InferenceSession(
        str(BASE_DIR / "vit-emotion-v2.onnx"), providers=["CPUExecutionProvider"]
    )

    classifier = fp.EmotionClassifier(
        session=session,
        input_name=session.get_inputs()[0].name,
        image_size=int(config.get("image_size", 224)),
        mean=np.array(config.get("mean", [0.5] * 3), dtype=np.float32),
        std=np.array(config.get("std", [0.5] * 3), dtype=np.float32),
        temperature=float(config.get("temperature", 1.0)),
        use_tta=False,
    )

    frame, box = synthetic_face_frame()
    crop = fp.crop_face(frame, box, fp.CropMode.ALIGNED, 224)

    probabilities = classifier.predict(crop)

    assert probabilities.shape == (NUM_SHARED_LABELS,)
    assert abs(probabilities.sum() - 1.0) < 1e-5
    assert np.all(probabilities >= 0.0)

    # TTA must not change the shape or break normalisation.
    classifier.use_tta = True
    with_tta = classifier.predict(crop)

    assert with_tta.shape == (NUM_SHARED_LABELS,)
    assert abs(with_tta.sum() - 1.0) < 1e-5


def test_temperature_softens_without_reordering():

    if not (BASE_DIR / "vit-emotion-v2.onnx").exists():
        raise Skip("ONNX model not present")

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(BASE_DIR / "vit-emotion-v2.onnx"), providers=["CPUExecutionProvider"]
    )

    frame, box = synthetic_face_frame()
    crop = fp.crop_face(frame, box, fp.CropMode.ALIGNED, 224)

    def build(temperature):
        return fp.EmotionClassifier(
            session=session,
            input_name=session.get_inputs()[0].name,
            image_size=224,
            mean=np.array([0.5] * 3, dtype=np.float32),
            std=np.array([0.5] * 3, dtype=np.float32),
            temperature=temperature,
            use_tta=False,
        )

    sharp = build(1.0).predict(crop)
    soft = build(2.5).predict(crop)

    assert sharp.argmax() == soft.argmax(), "scaling changed the prediction"
    assert soft.max() < sharp.max(), "higher temperature should soften"


# ============================================================
# THE AUDIO BUFFER
# ============================================================
#
# AudioBuffer only, not AudioEmotionRuntime: the runtime loads a
# 380 MB checkpoint, and the buffer is where the logic that can
# actually be wrong lives.
# ============================================================


def test_audio_buffer_reports_short_windows_as_none():
    """A window must not be scored before 4 seconds have arrived."""

    from audio_runtime import AudioBuffer

    buffer = AudioBuffer(capacity=32000)
    buffer.extend(np.zeros(4000, dtype=np.float32))

    assert buffer.window(16000) is None


def test_audio_buffer_returns_the_most_recent_samples():
    from audio_runtime import AudioBuffer

    buffer = AudioBuffer(capacity=32000)

    for start in range(0, 16000, 4000):
        buffer.extend(np.arange(start, start + 4000, dtype=np.float32))

    window = buffer.window(8000)

    # The newest 8000, not the oldest: the model is asked what the
    # speaker sounds like now.
    assert window is not None
    assert window[0] == 8000.0
    assert window[-1] == 15999.0


def test_audio_buffer_stays_bounded_over_a_long_session():
    """A session runs for minutes; the buffer must not grow with it."""

    from audio_runtime import AudioBuffer

    capacity = 32000
    buffer = AudioBuffer(capacity=capacity)

    for _ in range(500):
        buffer.extend(np.zeros(4000, dtype=np.float32))

    held = buffer.seconds_held * 16000

    # Eviction is whole-chunk, so a chunk's worth of slack is expected.
    assert held <= capacity + 4000, held
    assert buffer.window(16000) is not None


# ============================================================
# THE NEUTRAL MARGIN
# ============================================================


def margin_classifier(margin: float):
    """An EmotionClassifier with no ONNX session behind it.

    Only _apply_neutral_margin is under test, and it touches nothing
    but the probability vector, so constructing the real thing (and
    loading 343 MB of weights) would test the same arithmetic slower.
    """

    return fp.EmotionClassifier(
        session=None,
        input_name="pixel_values",
        image_size=224,
        mean=np.array([0.5] * 3, dtype=np.float32),
        std=np.array([0.5] * 3, dtype=np.float32),
        temperature=1.0,
        use_tta=False,
        neutral_margin=margin,
    )


def test_zero_margin_leaves_the_distribution_untouched():
    """The default must be a no-op, or every stored benchmark moves."""

    classifier = margin_classifier(0.0)

    before = peaked("surprise", 0.4)
    after = classifier._apply_neutral_margin(before)

    assert np.allclose(before, after), after


def test_margin_flips_a_near_tie_to_neutral():
    classifier = margin_classifier(0.15)

    # Neutral is the runner-up, 0.08 behind. Inside the margin.
    probabilities = np.array([0.02, 0.02, 0.02, 0.02, 0.40, 0.04, 0.48])

    adjusted = classifier._apply_neutral_margin(probabilities)

    assert SHARED_LABELS[int(np.argmax(probabilities))] == "surprise"
    assert SHARED_LABELS[int(np.argmax(adjusted))] == "neutral"


def test_margin_leaves_a_confident_rare_class_alone():
    """This is why the margin is additive in probability space.

    A log-prior correction strong enough to fix the neutral rate also
    drives disgust to never being predicted. Adding a constant only
    decides near-ties, so a confident disgust survives.
    """

    classifier = margin_classifier(0.20)

    probabilities = peaked("disgust", 0.75)
    adjusted = classifier._apply_neutral_margin(probabilities)

    assert SHARED_LABELS[int(np.argmax(adjusted))] == "disgust"


def test_margin_is_equivalent_to_the_runner_up_rule():
    """Adding m to neutral == 'pick neutral when within m of the top'.

    The equivalence is the whole justification for implementing it as a
    distribution transform rather than a post-hoc label override: it
    keeps the stored vector and the displayed label consistent, which
    matters because fusion pools the vector.
    """

    margin = 0.18
    classifier = margin_classifier(margin)
    generator = np.random.default_rng(0)

    neutral_index = SHARED_LABEL2ID["neutral"]

    for _ in range(500):
        probabilities = generator.dirichlet(np.ones(NUM_SHARED_LABELS))

        transformed = int(np.argmax(classifier._apply_neutral_margin(probabilities)))

        rule = (
            neutral_index
            if probabilities[neutral_index] >= probabilities.max() - margin
            else int(np.argmax(probabilities))
        )

        assert transformed == rule, (probabilities, transformed, rule)


def test_margin_output_is_still_a_distribution():
    classifier = margin_classifier(0.25)

    adjusted = classifier._apply_neutral_margin(peaked("fear", 0.5))

    assert abs(adjusted.sum() - 1.0) < 1e-9
    assert (adjusted >= 0).all()


# ============================================================
# THE AUDIO LEG
# ============================================================


def audio_evidence(
    vector: np.ndarray, voiced: float = 0.9, timestamp: float = 0.0
):
    return AudioEvidence(
        probabilities=vector, voiced_ratio=voiced, timestamp=timestamp
    )


def test_audio_alone_produces_an_audio_only_result():
    engine = FusionEngine()

    result = engine.fuse(
        None, 0.0, None, now=0.0,
        audio_evidence=audio_evidence(peaked("sad", 0.8)),
    )

    assert result.mode == "audio_only", result.mode
    assert result.label == "sad", result.label
    assert abs(result.audio.influence - 1.0) < 1e-9


def test_silent_window_contributes_nothing():
    """A window below the voiced floor must not vote at all."""

    engine = FusionEngine()

    # Audio claims 'happy' with high confidence, but the window is
    # 2% voiced - it is a pause, and the model's output for a pause
    # is not evidence about the speaker's emotion.
    result = engine.fuse(
        peaked("sad", 0.7), 0.9,
        None, now=0.0,
        audio_evidence=audio_evidence(peaked("happy", 0.95), voiced=0.02),
    )

    assert result.mode == "face_only", result.mode
    assert result.label == "sad", result.label
    assert result.audio.influence == 0.0


def test_stale_audio_decays_out_of_the_pool():
    engine = FusionEngine()

    fresh = engine.fuse(
        peaked("sad", 0.6), 0.9, None, now=0.0,
        audio_evidence=audio_evidence(peaked("happy", 0.9), timestamp=0.0),
    )

    # Past audio_cutoff_seconds the window contributes nothing and the
    # result must collapse cleanly to face-only.
    stale = engine.fuse(
        peaked("sad", 0.6), 0.9, None, now=120.0,
        audio_evidence=audio_evidence(peaked("happy", 0.9), timestamp=0.0),
    )

    assert fresh.mode == "fused", fresh.mode
    assert stale.mode == "face_only", stale.mode
    assert stale.label == "sad", stale.label


def test_three_modalities_all_contribute():
    engine = FusionEngine()

    result = engine.fuse(
        peaked("angry", 0.6), 0.9,
        evidence(peaked("angry", 0.6)),
        now=0.0,
        audio_evidence=audio_evidence(peaked("angry", 0.6)),
    )

    assert result.mode == "fused", result.mode
    assert result.label == "angry", result.label

    shares = [
        result.face.influence, result.audio.influence, result.text.influence
    ]

    assert all(share > 0 for share in shares), shares
    assert abs(sum(shares) - 1.0) < 1e-9, shares

    # Three agreeing modalities must be at least as confident as any
    # one of them alone - that is the whole point of pooling.
    assert result.confidence > 0.6, result.confidence


def test_two_against_one_outvotes_the_dissenter():
    """Audio joining face must be able to overturn a text-only call."""

    engine = FusionEngine()

    # The coalition has to be genuinely stronger than the dissenter,
    # not merely larger. Text carries the highest base weight of the
    # three, so a near-uniform face plus a lukewarm audio should NOT
    # overturn a confident message - and does not.
    face_and_text = engine.fuse(
        peaked("sad", 0.65), 0.9,
        evidence(peaked("happy", 0.80)),
        now=0.0,
    )

    with_audio = engine.fuse(
        peaked("sad", 0.65), 0.9,
        evidence(peaked("happy", 0.80)),
        now=0.0,
        audio_evidence=audio_evidence(peaked("sad", 0.80)),
    )

    assert face_and_text.label == "happy", face_and_text.label
    assert with_audio.label == "sad", with_audio.label


def test_one_dissenting_modality_raises_the_conflict_flag():
    engine = FusionEngine()

    # Face and audio agree on 'sad'; text is confidently 'happy'.
    # Averaging the three pairwise coefficients would dilute that
    # disagreement, so the flag is any() over pairs.
    result = engine.fuse(
        peaked("sad", 0.9), 0.9,
        evidence(peaked("happy", 0.9)),
        now=0.0,
        audio_evidence=audio_evidence(peaked("sad", 0.9)),
    )

    assert result.conflicted, result.agreement


def test_trimodal_payload_names_every_modality():
    engine = FusionEngine()

    payload = engine.fuse(
        peaked("fear", 0.5), 0.8,
        evidence(peaked("fear", 0.5)),
        now=0.0,
        audio_evidence=audio_evidence(peaked("fear", 0.5)),
    ).as_dict()

    assert set(payload["weights"]) == {"face", "audio", "text"}
    assert payload["audio"]["present"] is True
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-3


def test_two_modality_callers_are_unaffected():
    """The pre-audio call signature must behave exactly as before."""

    engine = FusionEngine()

    # now=0.0 against a timestamp of 0.0: without it the default is
    # wall-clock time.monotonic(), which makes the message hours old
    # and decays it straight out of the pool.
    result = engine.fuse(
        peaked("happy", 0.7), 0.9, evidence(peaked("happy", 0.7)), now=0.0
    )

    assert result.mode == "fused", result.mode
    assert result.audio.present is False
    assert result.audio.influence == 0.0
    assert abs(result.face.influence + result.text.influence - 1.0) < 1e-9


# ============================================================
# RUNNER
# ============================================================


def main() -> int:

    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    passed = failed = skipped = 0

    for name, function in tests:

        try:
            function()

        except Skip as reason:
            print(f"SKIP  {name}  ({reason})")
            skipped += 1

        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1

        else:
            print(f"PASS  {name}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
