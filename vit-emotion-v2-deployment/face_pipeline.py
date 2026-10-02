"""
Face modality: detect -> align -> assess quality -> classify -> smooth.

Separated from server.py so the same code path is exercised by the
WebSocket server and by eval_face.py. Any accuracy claim in the report
comes from this module running in one of its CropMode configurations,
not from a parallel implementation that might drift.

What the crop geometry is actually worth
----------------------------------------
An earlier revision of this module asserted that eye-line alignment was
the largest available accuracy lever. Measurement says otherwise, and
the correction is worth stating plainly because the opposite claim is
the intuitive one.

On FER2013 - the model's own distribution - a plain box crop beats an
aligned warp at every cell of the geometry grid:

    raw resize, no detector      71.9%
    detector + box crop          70.8%
    detector + aligned warp      66.6%   (best of 40 swept geometries)

The warp resamples an already small crop and replicates edge pixels
where the rotated frame leaves the source. On an upright face there is
no pose error to win back, so that cost is all there is.

The picture inverts once the head actually rolls. With a known roll
applied, aligned minus box, in accuracy points:

     0 deg -2.8    10 deg -3.3    20 deg +4.2    30 deg +8.7

So neither fixed policy is right. CropMode.ADAPTIVE measures the roll
from the eye line and pays for the warp only past ~18 degrees, where it
starts buying more than it costs.

The lesson generalises: the earlier claim came from comparing two
configurations on composited out-of-distribution images, where both
scored badly and the difference between them meant nothing. Evaluate on
the distribution the model was trained for before concluding anything
about preprocessing.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from labels import NUM_SHARED_LABELS, SHARED_LABELS


# ============================================================
# TUNABLES
# ============================================================
#
# Every constant a sweep might touch lives here, named, so
# eval_face.py can vary one at a time and the report can quote a
# number rather than a magic literal buried in a function.
# ============================================================

# --- alignment geometry ---
#
# The inter-ocular distance is the scale anchor because it is the most
# stable facial measurement under expression - brows, mouth and jaw all
# move with the emotion being classified, and the eye centres barely do.
#
# These two ratios are MEASURED, not reasoned about. An earlier revision
# guessed 0.42 / 0.40 from how FER crops are usually framed; a sweep over
# the grid (eval_face.py --sweep-crop) on the model's own test set showed
# that guess costing about 3 accuracy points against the best cell.
# Do not change them without re-running the sweep.

# Both framing constants were retuned after the original sweep was found
# to have been run against eval_data built from RAF-DB's train split -
# which is v3's own training data, so it measured memorisation. Re-swept
# on eval_data_unseen (JasonChen0317/FacialExpressions, trained on by
# neither model).
#
# The old values framed the face far tighter than either model was
# trained on, and the class that paid for it was 'neutral': at the old
# LEGACY_MARGIN v3 predicted neutral on 2 of 693 unseen webcam faces and
# scored F1 0.000 for it, while reporting ~90% confidence on calm faces
# it called 'angry'. Widening the frame costs nothing elsewhere and is
# most of what makes neutral reachable at all.
EYE_DISTANCE_RATIO = 0.24   # inter-ocular distance / output width (was 0.38)
EYE_Y_RATIO = 0.34          # eye-line height / output height
LEGACY_MARGIN = 0.25        # box padding (was 0.05)

# The model's answer moves with the framing more than it moves with the
# expression. Measured on one real webcam face, sweeping only the crop
# ratio and changing nothing else, the top label went
#
#   0.21 angry    0.24 neutral    0.27 disgust    0.30 disgust
#
# That is variance, not an opinion, and picking the ratio that happens
# to land on the right answer for one face is fitting to n=1. Averaging
# the logits over a spread of framings cuts the variance without
# needing to know the answer in advance, and it measured better on
# faces trained on by neither model: macro-F1 42.66 -> 44.31, neutral
# F1 40.16 -> 42.48 at equal detection cost.
#
# Four crops, one batched session.run. Flip augmentation on top of this
# measured slightly worse (neutral F1 39.65) so it stays off.
MULTICROP_RATIOS = (0.21, 0.24, 0.27, 0.30)

# Above this much head roll, warping is worth its resampling cost.
#
# Re-measured by eval_face.py --roll-test on eval_data_unseen/webcam
# (faces neither model trained on) under the deployed calibration,
# aligned crop minus plain box crop, accuracy points:
#
#      0 deg  -0.87     15 deg  +2.63
#      5 deg  +0.58     20 deg  +2.50
#     10 deg  +1.90     25 deg  +2.97
#                       30 deg  +2.84
#
# Warping always costs something - it resamples an already small crop
# and replicates edge pixels where the rotated frame runs off the
# source. On an upright face there is nothing to win back, so the cost
# is the whole story, and the box crop still wins at 0 degrees.
#
# The previous 18.0 came from a sweep on FER2013 scored at T=1 with no
# logit bias. That sweep put the crossover between 15 and 20 degrees
# and had the warp losing 3 points at 10. Neither holds for the
# pipeline actually served: a logit bias is not a monotonic rescaling,
# so it reorders predictions, and re-measured the crossover sits just
# under 10. Keeping 18.0 left the 10-18 band on the box crop, giving
# up roughly 2 points there for nothing.
#
# Hence ADAPTIVE: pay the cost only when there is something to buy.
ROLL_ALIGN_THRESHOLD_DEGREES = 10.0

# --- quality thresholds ---

MIN_FACE_WIDTH_RATIO = 0.22   # face width / frame width for full marks
MIN_FACE_PIXELS = 110         # below this the crop is upsampled mush
SHARPNESS_TARGET = 150.0      # variance of Laplacian on the 224 crop
BRIGHTNESS_RANGE = (45.0, 210.0)
MIN_CONTRAST = 22.0           # std of the grayscale crop
MAX_YAW_PROXY = 0.38          # |nose offset| / inter-ocular distance
MAX_ROLL_DEGREES = 35.0
QUALITY_FLOOR = 0.05          # per-factor floor, so one bad factor
                              # drags the product down without zeroing it

# Relative importance of each quality factor in the geometric mean.
QUALITY_WEIGHTS = {
    "detection": 1.0,
    "size": 1.0,
    "sharpness": 1.0,
    "exposure": 0.7,
    "pose": 1.3,      # the failure mode the ViT handles worst
    "stability": 0.5,
}

# --- temporal smoothing ---

SMOOTHING_TAU_SECONDS = 0.6   # time constant of the probability EMA
SWITCH_MARGIN = 0.05          # hysteresis before the shown label flips
MAX_SMOOTHING_GAP = 2.0       # a longer gap than this restarts the state

# --- detector ---

YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 50
HAAR_ASSUMED_SCORE = 0.55     # Haar reports no confidence; this is a
                              # deliberately mediocre stand-in so the
                              # fallback path is never trusted like YuNet


class CropMode(str, Enum):
    """How the face crop is produced.

    ADAPTIVE is the default and the only one worth deploying: it uses
    the plain box crop on an upright face and the aligned warp on a
    rolled one, because measurement says each wins in a different
    regime (see ROLL_ALIGN_THRESHOLD_DEGREES).

    LEGACY and ALIGNED are the two fixed policies. They are kept so
    eval_face.py can report a like-for-like comparison rather than a
    comparison against a remembered number.
    """

    LEGACY = "legacy"
    ALIGNED = "aligned"
    ADAPTIVE = "adaptive"


# ============================================================
# DETECTION RESULT
# ============================================================


@dataclass
class FaceBox:
    """One detected face in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float
    # Five points in image coords when the detector provides them:
    # [eye_left, eye_right, nose, mouth_left, mouth_right], ordered by
    # image x within each pair.
    landmarks: np.ndarray | None = None

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def iou(self, other: FaceBox | None) -> float:
        """Overlap with a previous box, used as a stability signal."""

        if other is None:
            return 1.0

        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h

        inter_w = max(0, min(ax2, bx2) - max(self.x, other.x))
        inter_h = max(0, min(ay2, by2) - max(self.y, other.y))
        intersection = inter_w * inter_h

        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


def _order_landmarks(points: np.ndarray) -> np.ndarray:
    """Normalise landmark order to [eye_l, eye_r, nose, mouth_l, mouth_r].

    YuNet labels its eye points by the subject's own left and right,
    which swap in image coordinates depending on head pose. Sorting each
    pair by image x makes the downstream geometry independent of that.
    """

    eyes = points[0:2][np.argsort(points[0:2, 0])]
    mouth = points[3:5][np.argsort(points[3:5, 0])]

    return np.vstack([eyes, points[2:3], mouth]).astype(np.float32)


# ============================================================
# DETECTORS
# ============================================================


class YuNetDetector:
    """OpenCV's YuNet: a confidence score and five landmarks per face.

    The landmarks are the reason this is preferred over Haar - without
    them there is no eye line, and without an eye line there is no
    alignment.
    """

    name = "yunet"
    provides_landmarks = True

    def __init__(self, model_path: Path):
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
        self._input_size: tuple[int, int] = (320, 320)

    def detect(self, frame_bgr: np.ndarray) -> FaceBox | None:
        height, width = frame_bgr.shape[:2]

        # setInputSize reallocates internal buffers, so only call it
        # when the resolution actually changes - a webcam stream holds
        # one size for its whole session.
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, detections = self._detector.detect(frame_bgr)

        if detections is None or len(detections) == 0:
            return None

        # Largest face rather than highest score: the subject is the
        # person at the camera, and a confident detection of someone in
        # the background is still the wrong face.
        best = max(detections, key=lambda row: row[2] * row[3])

        x, y, w, h = (int(round(float(value))) for value in best[:4])

        landmarks = np.array(best[4:14], dtype=np.float32).reshape(5, 2)

        return FaceBox(
            x=x,
            y=y,
            w=w,
            h=h,
            score=float(np.clip(best[14], 0.0, 1.0)),
            landmarks=_order_landmarks(landmarks),
        )


class HaarDetector:
    """Offline fallback. Loose boxes, no score, no landmarks.

    This is the configuration the original 67.7% was measured under. It
    is kept so the server still runs without the YuNet asset, not
    because it is good.
    """

    name = "haar"
    provides_landmarks = False

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self._cascade.empty():
            raise RuntimeError("Could not load the OpenCV Haar face detector.")

    def detect(self, frame_bgr: np.ndarray) -> FaceBox | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
        )

        if len(faces) == 0:
            # A second pass at 1.5x helps with faces close to the
            # camera, where the first scale search can step over them.
            enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5)
            found = self._cascade.detectMultiScale(
                enlarged, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40)
            )
            faces = [
                (int(x / 1.5), int(y / 1.5), int(w / 1.5), int(h / 1.5))
                for x, y, w, h in found
            ]

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

        return FaceBox(
            x=int(x),
            y=int(y),
            w=int(w),
            h=int(h),
            score=HAAR_ASSUMED_SCORE,
            landmarks=None,
        )


def build_detector(prefer_yunet: bool = True):
    """Prefer YuNet; fall back to Haar so the server always starts."""

    from setup_models import YUNET_PATH

    if prefer_yunet and hasattr(cv2, "FaceDetectorYN") and YUNET_PATH.exists():
        try:
            return YuNetDetector(YUNET_PATH)
        except cv2.error:
            pass

    return HaarDetector()


# ============================================================
# EXECUTION PROVIDERS
# ============================================================
#
# ONNX Runtime picks the first provider in the list it can actually
# initialise, so the order here is a preference ranking, not a
# requirement - a machine with no GPU silently lands on CPU.
#
# CUDA is tried before DirectML because it is meaningfully faster on
# NVIDIA hardware; DirectML is kept because it works on any DX12 GPU
# (including AMD and Intel) without a separate CUDA toolkit install,
# which on Windows is the difference between "pip install" and an
# afternoon.
# ============================================================

PROVIDER_PREFERENCE = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

_cuda_dlls_registered = False


def _register_cuda_dlls() -> None:
    """Point the Windows DLL loader at torch's bundled CUDA libraries.

    ONNX Runtime's CUDA provider needs cuDNN and cuBLAS on the search
    path. A CUDA-enabled torch already ships exactly those DLLs in
    `torch/lib`, so when torch is installed there is nothing to
    download - the loader just has to be told where to look. Without
    this the provider fails to initialise with "cublasLt64_13.dll is
    missing" and silently falls back to CPU.

    No-op off Windows, where the loader uses LD_LIBRARY_PATH instead,
    and harmless when torch is absent or CPU-only.
    """

    global _cuda_dlls_registered

    if _cuda_dlls_registered or not hasattr(os, "add_dll_directory"):
        return

    _cuda_dlls_registered = True

    try:
        import torch
    except ImportError:
        return

    lib = Path(torch.__file__).resolve().parent / "lib"

    if lib.is_dir():
        try:
            os.add_dll_directory(str(lib))
        except OSError:
            pass


def select_providers(prefer_gpu: bool = True) -> list[str]:
    """Rank the available ONNX Runtime providers, best first."""

    import onnxruntime as ort

    if not prefer_gpu:
        return ["CPUExecutionProvider"]

    _register_cuda_dlls()

    available = set(ort.get_available_providers())

    chosen = [name for name in PROVIDER_PREFERENCE if name in available]

    # CPU is always appended as the final fallback even if it somehow
    # did not appear in the preference list, so the session can never
    # fail to construct for want of a provider.
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")

    return chosen


def describe_providers(session: Any) -> str:
    """What the session is actually running on, for logs and /health."""

    active = session.get_providers()

    return active[0] if active else "unknown"


# ============================================================
# ALIGNMENT
# ============================================================


def _pseudo_landmarks(box: FaceBox) -> np.ndarray:
    """Approximate eye/mouth points from a bare bounding box.

    Used when the detector gives no landmarks. It cannot correct head
    roll - there is no measured eye line - but it does normalise scale
    and framing to the same canonical geometry the landmark path
    produces, which is most of the benefit. Ratios are the usual
    frontal-face proportions.
    """

    x, y, w, h = box.x, box.y, box.w, box.h

    return np.array(
        [
            [x + 0.30 * w, y + 0.38 * h],   # eye, image-left
            [x + 0.70 * w, y + 0.38 * h],   # eye, image-right
            [x + 0.50 * w, y + 0.55 * h],   # nose tip
            [x + 0.35 * w, y + 0.75 * h],   # mouth corner, image-left
            [x + 0.65 * w, y + 0.75 * h],   # mouth corner, image-right
        ],
        dtype=np.float32,
    )


def similarity_transform(
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    output_size: int,
    eye_distance_ratio: float | None = None,
) -> np.ndarray:
    """Closed-form 2x3 similarity mapping the eye line to canonical position.

    Two source points and two destination points determine a similarity
    exactly, so this is solved directly rather than through
    estimateAffinePartial2D - no RANSAC, no iteration, and byte-identical
    output for identical input, which matters for reproducible eval runs.
    """

    # Passed explicitly rather than read from the module global, so the
    # multi-crop ensemble can vary it without mutating shared state -
    # the server runs this on a threadpool.
    ratio = EYE_DISTANCE_RATIO if eye_distance_ratio is None else eye_distance_ratio

    destination_left = np.array(
        [
            (0.5 - ratio / 2.0) * output_size,
            EYE_Y_RATIO * output_size,
        ],
        dtype=np.float64,
    )
    destination_right = np.array(
        [
            (0.5 + ratio / 2.0) * output_size,
            EYE_Y_RATIO * output_size,
        ],
        dtype=np.float64,
    )

    source_left = np.asarray(eye_left, dtype=np.float64)
    source_right = np.asarray(eye_right, dtype=np.float64)

    source_vector = source_right - source_left
    destination_vector = destination_right - destination_left

    source_length = float(np.linalg.norm(source_vector))

    if source_length < 1e-3:
        raise ValueError("Degenerate eye line: the two eye points coincide.")

    scale = float(np.linalg.norm(destination_vector)) / source_length

    angle = math.atan2(
        destination_vector[1], destination_vector[0]
    ) - math.atan2(source_vector[1], source_vector[0])

    cos_a = scale * math.cos(angle)
    sin_a = scale * math.sin(angle)

    # translation = destination_left - (R * s) @ source_left
    translation_x = destination_left[0] - (
        cos_a * source_left[0] - sin_a * source_left[1]
    )
    translation_y = destination_left[1] - (
        sin_a * source_left[0] + cos_a * source_left[1]
    )

    return np.array(
        [
            [cos_a, -sin_a, translation_x],
            [sin_a, cos_a, translation_y],
        ],
        dtype=np.float32,
    )


def _legacy_crop(
    frame_bgr: np.ndarray,
    box: FaceBox,
    output_size: int,
) -> np.ndarray | None:
    """The original path: box plus a 5% margin, resized. No alignment."""

    frame_h, frame_w = frame_bgr.shape[:2]

    margin_x = int(box.w * LEGACY_MARGIN)
    margin_y = int(box.h * LEGACY_MARGIN)

    x1 = max(0, box.x - margin_x)
    y1 = max(0, box.y - margin_y)
    x2 = min(frame_w, box.x + box.w + margin_x)
    y2 = min(frame_h, box.y + box.h + margin_y)

    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (output_size, output_size))


def roll_degrees(box: FaceBox) -> float:
    """Head roll in degrees from the eye line, 0 when it is horizontal.

    Returns 0.0 without landmarks, which routes the adaptive mode to
    the plain box crop - the right default, since a detector that gives
    no landmarks also gives no way to warp accurately.
    """

    if box.landmarks is None:
        return 0.0

    vector = box.landmarks[1].astype(np.float64) - box.landmarks[0].astype(np.float64)

    if float(np.linalg.norm(vector)) < 1e-3:
        return 0.0

    angle = abs(math.degrees(math.atan2(vector[1], vector[0])))

    # Fold onto [0, 90]: a 170 degree eye line is a 10 degree roll with
    # the landmarks labelled the other way round.
    return min(angle, 180.0 - angle)


def crop_face(
    frame_bgr: np.ndarray,
    box: FaceBox,
    mode: CropMode,
    output_size: int,
    eye_distance_ratio: float | None = None,
) -> np.ndarray | None:
    """Produce the model-ready face crop for the requested mode."""

    if mode is CropMode.ADAPTIVE:
        mode = (
            CropMode.ALIGNED
            if roll_degrees(box) >= ROLL_ALIGN_THRESHOLD_DEGREES
            else CropMode.LEGACY
        )

    if mode is CropMode.LEGACY:
        return _legacy_crop(frame_bgr, box, output_size)

    landmarks = box.landmarks
    if landmarks is None:
        landmarks = _pseudo_landmarks(box)

    try:
        matrix = similarity_transform(
            landmarks[0], landmarks[1], output_size, eye_distance_ratio
        )
    except ValueError:
        return _legacy_crop(frame_bgr, box, output_size)

    # BORDER_REPLICATE rather than a constant fill: a rolled or
    # off-centre head warps part of the output off the source image, and
    # black bars are far outside anything the ViT saw in training. Edge
    # replication is also wrong, but wrong in a way the model tolerates.
    return cv2.warpAffine(
        frame_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ============================================================
# QUALITY
# ============================================================


@dataclass
class Quality:
    """Per-frame reliability of the face evidence, factor by factor.

    `overall` is what fusion consumes; the named sub-scores are what the
    UI shows, so a low weight is explainable instead of arbitrary.
    """

    detection: float
    size: float
    sharpness: float
    exposure: float
    pose: float
    stability: float
    overall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "detection": round(self.detection, 4),
            "size": round(self.size, 4),
            "sharpness": round(self.sharpness, 4),
            "exposure": round(self.exposure, 4),
            "pose": round(self.pose, 4),
            "stability": round(self.stability, 4),
            "overall": round(self.overall, 4),
            "weakest": self.weakest(),
        }

    @classmethod
    def absent(cls) -> Quality:
        """No face: every factor zero, and a hard zero overall."""

        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def weakest(self) -> str:
        """Name of the factor currently costing the most, for the UI."""

        factors = {
            "detection": self.detection,
            "size": self.size,
            "sharpness": self.sharpness,
            "exposure": self.exposure,
            "pose": self.pose,
            "stability": self.stability,
        }

        return min(factors, key=lambda key: factors[key])


def _ramp(value: float, target: float) -> float:
    """Linear 0..1 ramp that saturates at `target`."""

    if target <= 0:
        return 1.0

    return float(np.clip(value / target, 0.0, 1.0))


def _pose_score(box: FaceBox) -> float:
    """Frontality, from the nose's offset against the eye line.

    A frontal face puts the nose tip near the midpoint of the eye line;
    yaw slides it toward one eye. This is a proxy, not a pose estimate,
    but it needs no extra model and it tracks the failure mode that
    matters: profile faces the ViT never saw in training.

    Roll is only mildly penalised because the aligned crop corrects it
    outright; what remains is the resampling cost of a large rotation.
    """

    if box.landmarks is None:
        # No landmarks means no pose information. Return a middling
        # score rather than a good one - the Haar path genuinely does
        # not know whether the face is turned away.
        return 0.6

    eye_left = box.landmarks[0].astype(np.float64)
    eye_right = box.landmarks[1].astype(np.float64)
    nose = box.landmarks[2].astype(np.float64)

    eye_vector = eye_right - eye_left
    inter_ocular = float(np.linalg.norm(eye_vector))

    if inter_ocular < 1e-3:
        return 0.0

    eye_midpoint = (eye_left + eye_right) / 2.0

    # Offset measured along the eye line, so it stays meaningful when
    # the head is also rolled.
    unit = eye_vector / inter_ocular
    yaw_proxy = abs(float(np.dot(nose - eye_midpoint, unit))) / inter_ocular

    yaw_score = float(np.clip(1.0 - yaw_proxy / MAX_YAW_PROXY, 0.0, 1.0))

    roll_degrees = abs(math.degrees(math.atan2(eye_vector[1], eye_vector[0])))
    roll_degrees = min(roll_degrees, 180.0 - roll_degrees)
    roll_score = float(
        np.clip(1.0 - roll_degrees / (MAX_ROLL_DEGREES * 2.0), 0.3, 1.0)
    )

    return yaw_score * roll_score


def assess_quality(
    frame_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    box: FaceBox,
    previous_box: FaceBox | None,
) -> Quality:
    """Score how much this frame's face evidence deserves to be trusted."""

    frame_w = frame_bgr.shape[1]

    # --- detector confidence ---
    detection = float(np.clip(box.score, 0.0, 1.0))

    # --- size: both relative framing and absolute pixels ---
    # A face can fill the frame and still be unusable if the frame
    # itself is 160px wide, so both conditions are required.
    size = min(
        _ramp(box.w / max(frame_w, 1), MIN_FACE_WIDTH_RATIO),
        _ramp(float(box.w), MIN_FACE_PIXELS),
    )

    # --- sharpness, measured on the resized crop so it is scale-free ---
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = _ramp(
        float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()),
        SHARPNESS_TARGET,
    )

    # --- exposure: brightness inside a usable band, plus contrast ---
    brightness = float(gray_crop.mean())
    contrast = float(gray_crop.std())

    low, high = BRIGHTNESS_RANGE
    midpoint = (low + high) / 2.0
    half_span = (high - low) / 2.0

    brightness_score = float(
        np.clip(1.0 - abs(brightness - midpoint) / half_span, 0.0, 1.0)
    )
    exposure = min(brightness_score, _ramp(contrast, MIN_CONTRAST))

    # --- pose ---
    pose = _pose_score(box)

    # --- temporal stability ---
    stability = float(np.clip(box.iou(previous_box), 0.0, 1.0))

    values = {
        "detection": detection,
        "size": size,
        "sharpness": sharpness,
        "exposure": exposure,
        "pose": pose,
        "stability": stability,
    }

    # Weighted geometric mean. A product is the right shape here: these
    # are closer to independent necessary conditions than to
    # interchangeable contributions, so a face that is sharp, bright and
    # large but turned 60 degrees away should still score low. An
    # arithmetic mean would let the three good factors carry it.
    log_sum = sum(
        QUALITY_WEIGHTS[name] * math.log(max(value, QUALITY_FLOOR))
        for name, value in values.items()
    )
    weight_sum = sum(QUALITY_WEIGHTS[name] for name in values)

    overall = float(math.exp(log_sum / weight_sum))

    return Quality(overall=overall, **values)


# ============================================================
# CLASSIFIER
# ============================================================


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)

    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


class EmotionClassifier:
    """The ViT ONNX head, with test-time augmentation and calibration."""

    def __init__(
        self,
        session: Any,
        input_name: str,
        image_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        temperature: float = 1.0,
        logit_bias: Sequence[float] | None = None,
        use_tta: bool = True,
    ):
        self.session = session
        self.input_name = input_name
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Temperature > 1 softens the distribution. The raw ViT is
        # overconfident, and an overconfident modality would dominate
        # the fusion pool through the certainty term alone. Fitted by
        # eval_face.py and stored in emotion_config.json.
        self.temperature = max(float(temperature), 1e-3)

        self.logit_bias = (
            np.asarray(logit_bias, dtype=np.float32)
            if logit_bias is not None
            else np.zeros(NUM_SHARED_LABELS, dtype=np.float32)
        )

        self.use_tta = use_tta

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> normalised CHW float32 tensor."""

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        if face_rgb.shape[:2] != (self.image_size, self.image_size):
            face_rgb = cv2.resize(face_rgb, (self.image_size, self.image_size))

        image = face_rgb.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        return image.transpose(2, 0, 1)

    def logits_many(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Mean uncalibrated logits over several views of one face.

        One batched session.run for the whole ensemble - the ONNX graph
        takes a dynamic batch, so four crops cost far less than four
        calls. Averaging happens in logit space, before temperature and
        bias, so the stored calibration still applies to the result.
        """

        usable = [c for c in crops if c is not None and c.size]

        if not usable:
            raise ValueError("no usable crops")

        batch = np.stack([self.preprocess(c) for c in usable]).astype(np.float32)

        return self.session.run(None, {self.input_name: batch})[0].mean(axis=0)

    def predict_many(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Calibrated probabilities from a multi-crop ensemble."""

        return softmax(self.logits_many(crops) / self.temperature + self.logit_bias)

    def logits(self, face_bgr: np.ndarray) -> np.ndarray:
        """Uncalibrated averaged logits, exposed for temperature fitting."""

        tensors = [self.preprocess(face_bgr)]

        if self.use_tta:
            # Horizontal mirror. Faces are near-symmetric and every
            # shared class is symmetric too, so the mirrored crop is a
            # legitimate second view of the same expression rather than
            # a different expression. Averaging the two logit vectors
            # cancels some of the left/right lighting and pose
            # asymmetry the model is sensitive to.
            tensors.append(self.preprocess(cv2.flip(face_bgr, 1)))

        batch = np.stack(tensors).astype(np.float32)

        # One session.run for both views: the ONNX graph takes a
        # dynamic batch, and two forward passes in one call cost
        # noticeably less than two separate calls.
        outputs = self.session.run(None, {self.input_name: batch})[0]

        return outputs.mean(axis=0)

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return the calibrated 7-class probability vector."""

        calibrated = self.logits(face_bgr) / self.temperature + self.logit_bias

        return softmax(calibrated)


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================


class ProbabilitySmoother:
    """Quality-weighted, time-aware EMA over the probability vector.

    Two departures from a plain EMA, both deliberate:

    * The step is computed from elapsed time, not frame count. Frames
      arrive irregularly - the client paces them on server acks - so a
      fixed per-frame alpha would make the effective smoothing window
      depend on how fast the machine happens to be running.

    * The step is scaled by frame quality. A blurry, badly-posed frame
      should nudge the state; a clean frontal one should move it. That
      is what stops a single bad frame from flipping the displayed
      emotion, without the lag a uniformly heavy filter would add.
    """

    def __init__(
        self,
        tau_seconds: float = SMOOTHING_TAU_SECONDS,
        switch_margin: float = SWITCH_MARGIN,
    ):
        self.tau = max(tau_seconds, 1e-3)
        self.switch_margin = switch_margin

        self.state: np.ndarray | None = None
        self.label_index: int | None = None
        self._last_update: float | None = None

    def reset(self) -> None:
        self.state = None
        self.label_index = None
        self._last_update = None

    def update(
        self,
        probabilities: np.ndarray,
        quality: float,
        now: float | None = None,
    ) -> np.ndarray:
        """Fold one observation in and return the smoothed distribution."""

        now = time.monotonic() if now is None else now

        if self.state is None or self._last_update is None:
            self.state = probabilities.astype(np.float64).copy()
            self._last_update = now
            self.label_index = int(np.argmax(self.state))
            return self.state

        elapsed = now - self._last_update
        self._last_update = now

        if elapsed > MAX_SMOOTHING_GAP:
            # The stream stalled. Whatever the subject's face was doing
            # before the gap is no longer evidence about now.
            self.state = probabilities.astype(np.float64).copy()
            self.label_index = int(np.argmax(self.state))
            return self.state

        # Exponential step for the elapsed interval, scaled by quality.
        alpha = (1.0 - math.exp(-max(elapsed, 0.0) / self.tau)) * float(
            np.clip(quality, 0.0, 1.0)
        )

        self.state = (1.0 - alpha) * self.state + alpha * probabilities

        total = float(self.state.sum())
        if total > 0:
            self.state /= total

        self._apply_hysteresis()

        return self.state

    def _apply_hysteresis(self) -> None:
        """Require a margin before the reported label changes.

        Without this the label flickers whenever two classes sit within
        noise of each other - very common between sad/neutral and
        angry/disgust, which is exactly where this ViT is weakest.
        """

        assert self.state is not None

        leader = int(np.argmax(self.state))

        if self.label_index is None:
            self.label_index = leader
            return

        if leader == self.label_index:
            return

        margin = self.state[leader] - self.state[self.label_index]

        if margin >= self.switch_margin:
            self.label_index = leader

    @property
    def label(self) -> str | None:
        if self.label_index is None:
            return None

        return SHARED_LABELS[self.label_index]


# ============================================================
# ORCHESTRATOR
# ============================================================


@dataclass
class FaceResult:
    """Everything one frame produced, ready to serialise."""

    detected: bool
    box: FaceBox | None = None
    quality: Quality = field(default_factory=Quality.absent)
    raw_probabilities: np.ndarray | None = None
    smoothed_probabilities: np.ndarray | None = None
    label: str | None = None
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "detected": self.detected,
            "bbox": self.box.as_dict() if self.box else None,
            "detector_confidence": (
                round(self.box.score, 4) if self.box else None
            ),
            "quality": self.quality.as_dict(),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probs": None,
        }

        if self.smoothed_probabilities is not None:
            payload["probs"] = {
                label: round(float(self.smoothed_probabilities[index]), 4)
                for index, label in enumerate(SHARED_LABELS)
            }

        if self.raw_probabilities is not None:
            payload["raw_probs"] = {
                label: round(float(self.raw_probabilities[index]), 4)
                for index, label in enumerate(SHARED_LABELS)
            }

        return payload


@dataclass
class FaceSessionState:
    """Mutable per-connection face state."""

    smoother: ProbabilitySmoother = field(default_factory=ProbabilitySmoother)
    previous_box: FaceBox | None = None


class FacePipeline:
    """Runs the face path. Stateless across connections.

    The detector and ONNX session are safe to share between
    connections; the smoother and previous-box reference describe one
    person in front of one camera, so each connection gets its own via
    `session_state()`.
    """

    def __init__(
        self,
        detector: YuNetDetector | HaarDetector,
        classifier: EmotionClassifier,
        crop_mode: CropMode = CropMode.ALIGNED,
        multicrop: bool = True,
    ):
        self.detector = detector
        self.classifier = classifier
        self.crop_mode = crop_mode
        self.multicrop = multicrop

    def session_state(self) -> FaceSessionState:
        return FaceSessionState()

    def process(
        self,
        frame_bgr: np.ndarray,
        state: FaceSessionState,
        now: float | None = None,
    ) -> FaceResult:
        """Run the full face path for one frame."""

        box = self.detector.detect(frame_bgr)

        if box is None:
            state.previous_box = None
            state.smoother.reset()
            return FaceResult(detected=False)

        size = self.classifier.image_size

        crop = crop_face(frame_bgr, box, self.crop_mode, size)

        if crop is None or crop.size == 0:
            state.previous_box = None
            state.smoother.reset()
            return FaceResult(detected=False)

        # Quality is scored on the primary crop so the number keeps
        # meaning the same thing it did before the ensemble existed.
        quality = assess_quality(frame_bgr, crop, box, state.previous_box)
        state.previous_box = box

        if self.multicrop and self.crop_mode is not CropMode.LEGACY:
            views = [
                crop_face(frame_bgr, box, CropMode.ALIGNED, size,
                          eye_distance_ratio=ratio)
                for ratio in MULTICROP_RATIOS
            ]
            views = [v for v in views if v is not None and v.size]
            raw = (self.classifier.predict_many(views) if views
                   else self.classifier.predict(crop))
        else:
            raw = self.classifier.predict(crop)

        smoothed = state.smoother.update(raw, quality.overall, now=now)

        index = state.smoother.label_index
        index = 0 if index is None else index

        return FaceResult(
            detected=True,
            box=box,
            quality=quality,
            raw_probabilities=raw,
            smoothed_probabilities=smoothed.copy(),
            label=SHARED_LABELS[index],
            confidence=float(smoothed[index]),
        )
