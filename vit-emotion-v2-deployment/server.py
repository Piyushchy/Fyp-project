"""
FastAPI backend for real-time multimodal emotion recognition.

One WebSocket carries both modalities. Video frames arrive as
Base64-encoded JPEG and run face detection -> alignment -> ViT ONNX
inference -> temporal smoothing. Text messages arrive on the same
socket and run a fine-tuned compact transformer. Every frame reports
the fused result alongside each modality's own opinion.

Usage:
    python setup_models.py          # one-time, fetches the detector
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/ - the page is served from here rather
than opened as a file:// URL, because the client derives the WebSocket
address from window.location and file:// has no hostname to derive it
from.

Protocol
--------
Client -> server, either:
    {"type": "frame", "data": "<base64 jpeg>"}
    {"type": "text",  "text": "a message", "id": 7}
A bare Base64 string is still accepted and treated as a frame, so the
original single-modality client keeps working.

Server -> client:
    {"type": "frame", fps, face{...}, text{...}, fusion{...}}
    {"type": "text_result", id, label, confidence, probs{...}}
    {"type": "error", message}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from face_pipeline import (
    CropMode,
    EmotionClassifier,
    FacePipeline,
    FaceSessionState,
    build_detector,
    describe_providers,
    select_providers,
)
from fusion import FusionEngine, TextEvidence
from labels import SHARED_LABELS, assert_matches_config

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Which checkpoint to serve. See ../training/README.md.
#
#   v2  FER2013 only. Best on FER2013's own test split, worst on faces
#       from anywhere else.
#   v3  + AffectNet + RAF-DB. Better cross-dataset than v2, but trained
#       on a RAF-DB 'angry' folder whose images draw more 'neutral'
#       votes than 'angry' ones, which taught it to answer 'angry' for
#       a calm face.
#   v4  + FERPlus (FER2013 re-annotated by 10 crowd workers), with
#       RAF-DB's angry/disgust dropped as mostly label noise, and
#       trained with randomised framing so it does not depend on one
#       exact crop.
#
# Weights and config travel together - pairing v2's weights with v3's
# config is a bug the eval harness actually shipped with, so the mapping
# lives in one dict rather than two parallel ternaries.
MODELS = {
    "v2": ("vit-emotion-v2.onnx", "emotion_config.v2.json"),
    "v3": ("vit-emotion-v3.onnx", "emotion_config.json"),
    "v4": ("vit-emotion-v4.onnx", "emotion_config.v4.json"),
    "v5": ("vit-emotion-v5.onnx", "emotion_config.v5.json"),
    "v6": ("vit-emotion-v6.onnx", "emotion_config.v6.json"),
}

MODEL_KEY = os.environ.get("EMOTION_MODEL", "v4").lower()

if MODEL_KEY not in MODELS:
    raise SystemExit(
        f"EMOTION_MODEL={MODEL_KEY!r} is not one of {sorted(MODELS)}"
    )

_ONNX_NAME, _CONFIG_NAME = MODELS[MODEL_KEY]

ONNX_PATH = BASE_DIR / _ONNX_NAME
CONFIG_PATH = BASE_DIR / _CONFIG_NAME
CONFIG_PATH_V2 = BASE_DIR / "emotion_config.v2.json"
INDEX_PATH = BASE_DIR / "index.html"

# The text module lives in a sibling directory whose name contains
# hyphens, so it cannot be imported as a package. Its modules use flat
# imports among themselves (`from config import ...`), which is why the
# directory itself goes on sys.path rather than its parent.
TEXT_MODULE_DIR = BASE_DIR.parent / "text-emotion-module"


# ============================================================
# SETTINGS
# ============================================================

# Which text checkpoint to serve, in order of preference. GoEmotions
# first because it predicts directly in the shared seven-class space;
# the MTEB checkpoints are a usable fallback but cannot express
# 'neutral' or 'disgust', which is a real handicap for fusion.
TEXT_MODEL_PREFERENCE = [
    ("goemotions", "tinybert"),
    ("goemotions", "mobilebert"),
    ("goemotions", "distilbert"),
    ("mteb", "tinybert"),
]

# Flip TTA roughly doubles ONNX cost. Measured on FER2013 it is worth
# well under a point either way, so it is off by default; turn it on if
# you have the frame budget and want the small variance reduction.
USE_TTA = False

# ============================================================
# NEUTRAL MARGIN
# ============================================================
#
# Probability mass added to 'neutral' before the argmax. It fixes
# a prior mismatch, not a modelling error: the calibration set is
# balanced, a webcam session is roughly 55% neutral, and a model
# carries whatever prior it was calibrated under. Measured on
# eval_data_unseen/test/webcam under a realistic session prior,
# neutral is shown 27.5% of the time against a true 55%, while
# surprise runs at 2.08x its real rate and sad at 2.24x - and
# because most frames are neutral, nearly every false 'surprise'
# a user sees is a neutral face that leaked.
#
#   0.00  off. The default, and what every number in
#         results_face_eval.*.json is measured at.
#   0.15  neutral shown 39.7%, surprise 1.82x, balanced macro-F1
#         0.4738 (against 0.5027 off)
#   0.20  neutral shown 45.8%, surprise 1.62x, balanced macro-F1
#         0.4669
#
# Off by default because it is a serving-time decision rule and
# should not silently change what the evaluation harness reports.
# Set EMOTION_NEUTRAL_MARGIN=0.18 to turn it on. See
# diagnose_live_prior.py for the measurement and for why this
# beats the textbook log-prior correction.
# ============================================================

NEUTRAL_MARGIN = float(os.environ.get("EMOTION_NEUTRAL_MARGIN", "0.0"))

if not 0.0 <= NEUTRAL_MARGIN < 1.0:
    raise SystemExit(
        f"EMOTION_NEUTRAL_MARGIN must be in [0, 1), got {NEUTRAL_MARGIN}"
    )

# Use the GPU when one is available for it. Both modalities honour this.
# Set EMOTION_DEVICE=cpu to force CPU, which is worth doing when
# comparing latency numbers against the CPU-only figures in the README.
#   auto  use the GPU if a provider registers, fall back to CPU quietly
#   gpu   same, but refuse to start on CPU - a silent fallback is how
#         this project ran a 46x slowdown unnoticed, so asking for the
#         GPU explicitly means you want to be told when you did not get
#         it rather than discover it from the frame rate
#   cpu   force CPU
DEVICE = os.environ.get("EMOTION_DEVICE", "auto").lower()

if DEVICE not in ("auto", "gpu", "cpu"):
    raise SystemExit(
        f"EMOTION_DEVICE={DEVICE!r} is not one of 'auto', 'gpu', 'cpu'"
    )

PREFER_GPU = DEVICE != "cpu"
REQUIRE_GPU = DEVICE == "gpu"


def _onnxruntime_distributions() -> set[str]:
    """Which onnxruntime wheels are installed.

    More than one is a misconfiguration rather than a choice: they all
    unpack into the same `onnxruntime` package, so the survivor is
    whichever pip touched last, and the loser's execution providers go
    missing without raising anything.
    """

    from importlib import metadata

    known = {
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
    }

    found = set()

    for distribution in metadata.distributions():
        name = (distribution.metadata["Name"] or "").lower()
        if name in known:
            found.add(name)

    return found


def torch_device() -> str:
    """Which device the text model should live on.

    A CPU-only torch build reports no CUDA, so this returns "cpu"
    without the caller needing to know which wheel is installed.
    """

    if not PREFER_GPU:
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    # Apple silicon, for completeness.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"

# Always align, for v4. The eye-line warp normalises framing against
# the inter-ocular distance, which is stable; the box crop inherits
# however tight the detector's box happened to be that frame, and that
# variance is what the model is most sensitive to. Measured on faces
# trained on by neither model, aligned matched the box crop on accuracy
# and beat it where it mattered: neutral F1 32.8% against 21.6%.
#
# ADAPTIVE (box below a roll threshold, aligned above) was the previous
# default and was tuned against an eval set drawn from v3's own
# training data.
CROP_MODE = CropMode.ALIGNED

# Average the logits over several crop ratios. See MULTICROP_RATIOS.
USE_MULTICROP = True

MAX_TEXT_LENGTH = 2000


# ============================================================
# GLOBAL SINGLETONS
# ============================================================


@dataclass
class Runtime:
    """Everything loaded once at startup and shared across connections.

    All three are safe to share: the ONNX session and the detector are
    stateless per call, and the text classifier is only read from. The
    mutable per-person state lives in SessionState instead.
    """

    face: FacePipeline | None = None
    fusion: FusionEngine | None = None
    text_classifier: Any = None
    text_model_name: str = "unavailable"
    text_task: str = ""
    detector_name: str = ""
    face_provider: str = "unknown"
    text_device: str = "cpu"


runtime = Runtime()


@dataclass
class SessionState:
    """Mutable state belonging to one WebSocket connection.

    Kept per-connection rather than global so two browsers pointed at
    the same server do not smooth each other's faces together or read
    each other's messages.
    """

    face: FaceSessionState
    text: TextEvidence | None = None
    previous_frame_time: float = field(default_factory=time.monotonic)


# ============================================================
# TEXT MODEL LOADING
# ============================================================


def load_text_classifier(device: str = "cpu") -> tuple[Any, str, str]:
    """Load the best available text checkpoint.

    Returns (classifier, display_name, task_key). A missing checkpoint
    is not fatal - the server degrades to face-only rather than
    refusing to start, because the face path is independently useful
    and training the text model takes half an hour of CPU.
    """

    if str(TEXT_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(TEXT_MODULE_DIR))

    try:
        from predict import TextEmotionClassifier
    except ImportError as error:
        logger.warning("Text module unavailable (%s); running face-only.", error)
        return None, "unavailable", ""

    for task_key, model_key in TEXT_MODEL_PREFERENCE:
        try:
            classifier = TextEmotionClassifier(
                model_key=model_key, device=device, task=task_key
            )
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("Failed to load %s/%s", task_key, model_key)
            continue

        name = f"{classifier.display_name} [{task_key}]"

        logger.info("Text model loaded: %s (T=%.3f)", name, classifier.temperature)

        if task_key != "goemotions":
            logger.warning(
                "Using a 6-class checkpoint. It cannot predict 'neutral' or "
                "'disgust', so fusion will never see text evidence for "
                "either. Train the 7-class model with: "
                "python train.py --task goemotions --model tinybert"
            )

        return classifier, name, task_key

    logger.warning(
        "No text checkpoint found; running face-only. Train one with: "
        "python train.py --task goemotions --model tinybert"
    )

    return None, "unavailable", ""


# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load heavy resources once at startup; release on shutdown."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    id2label = {int(key): value for key, value in config["id2label"].items()}

    # Fail fast if the exported model's class order ever stops matching
    # the shared space. Fusion indexes both modalities by position, so
    # a mismatch would compare different emotions against each other
    # and nothing downstream could detect it.
    assert_matches_config(id2label)

    providers = select_providers(PREFER_GPU)

    logger.info("Loading ONNX model from %s (providers: %s)",
                ONNX_PATH, ", ".join(providers))

    session = ort.InferenceSession(str(ONNX_PATH), providers=providers)

    runtime.face_provider = describe_providers(session)

    if runtime.face_provider == "CPUExecutionProvider" and PREFER_GPU:
        # This used to be logged at info level, which is how an
        # accidental CPU fallback ran unnoticed at 0.8 FPS on a machine
        # with a working CUDA GPU. A GPU was asked for and not obtained:
        # that is a warning, and it names the likely cause.
        installed = _onnxruntime_distributions()

        if REQUIRE_GPU:
            raise SystemExit(
                "EMOTION_DEVICE=gpu was requested but no GPU execution "
                "provider registered, so every frame would run on CPU "
                "(~303 ms/frame against ~6.6 ms on CUDA).\n"
                f"  installed onnxruntime wheels: "
                f"{', '.join(sorted(installed)) or 'none'}\n"
                f"  available providers: "
                f"{', '.join(ort.get_available_providers())}\n"
                + (
                    "  Both wheels are installed; they share one package "
                    "directory so the last installed overwrites the "
                    "other's native module. Fix:\n"
                    "    pip uninstall -y onnxruntime onnxruntime-gpu && "
                    "pip install onnxruntime-gpu\n"
                    if len(installed) > 1 else
                    "  Install a GPU wheel: pip install onnxruntime-gpu "
                    "(NVIDIA/CUDA) or onnxruntime-directml (any DX12 GPU).\n"
                )
                + "  Or start with --device auto to allow the CPU fallback."
            )

        if len(installed) > 1:
            logger.warning(
                "Face inference fell back to CPU: %s are BOTH installed. "
                "They share one package directory, so the last one "
                "installed overwrites the other's native module and the "
                "CUDA provider silently disappears. Fix with: pip "
                "uninstall -y onnxruntime onnxruntime-gpu && pip install "
                "onnxruntime-gpu",
                " and ".join(sorted(installed)),
            )
        else:
            logger.warning(
                "Face inference fell back to CPU despite EMOTION_DEVICE=%s. "
                "Available providers: %s. For GPU: pip install "
                "onnxruntime-gpu (NVIDIA/CUDA) or onnxruntime-directml "
                "(any DX12 GPU). Expect roughly 6 ms/frame on GPU against "
                "300 ms/frame here.",
                os.environ.get("EMOTION_DEVICE", "auto"),
                ", ".join(ort.get_available_providers()),
            )
    else:
        logger.info("Face inference provider: %s", runtime.face_provider)

    classifier = EmotionClassifier(
        session=session,
        input_name=session.get_inputs()[0].name,
        image_size=int(config.get("image_size", 224)),
        mean=np.array(config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32),
        std=np.array(config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32),
        temperature=float(config.get("temperature", 1.0)),
        logit_bias=config.get("logit_bias"),
        use_tta=USE_TTA,
        neutral_margin=NEUTRAL_MARGIN,
    )

    if NEUTRAL_MARGIN > 0:
        logger.info(
            "Neutral margin: %.3f (corrects the balanced-calibration "
            "prior for a mostly-neutral session)",
            NEUTRAL_MARGIN,
        )

    detector = build_detector()

    runtime.face = FacePipeline(
        detector, classifier, crop_mode=CROP_MODE, multicrop=USE_MULTICROP
    )
    runtime.detector_name = detector.name
    runtime.fusion = FusionEngine()

    if detector.name == "haar":
        logger.warning(
            "Falling back to the Haar cascade: no landmarks, so crops "
            "cannot be aligned and accuracy will be materially lower. "
            "Run 'python setup_models.py' to fetch the YuNet detector."
        )
    else:
        logger.info("Detector: %s (aligned crops enabled)", detector.name)

    logger.info(
        "Face classifier ready - TTA=%s, crop=%s, T=%.3f",
        USE_TTA, CROP_MODE.value, classifier.temperature,
    )

    runtime.text_device = torch_device()

    (
        runtime.text_classifier,
        runtime.text_model_name,
        runtime.text_task,
    ) = load_text_classifier(runtime.text_device)

    logger.info("Text inference device: %s", runtime.text_device)

    yield

    runtime.face = None
    runtime.fusion = None
    runtime.text_classifier = None

    logger.info("Resources released")


app = FastAPI(
    title="Multimodal Emotion Recognition API",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# TEXT INFERENCE
# ============================================================


def classify_text(message: str) -> dict[str, Any] | None:
    """Run the text model and return a shared-space result."""

    if runtime.text_classifier is None:
        return None

    prediction = runtime.text_classifier.predict(message)

    shared = prediction["shared_distribution"]

    vector = np.array(
        [shared[label] for label in SHARED_LABELS], dtype=np.float64
    )

    total = vector.sum()
    if total > 0:
        vector = vector / total

    index = int(np.argmax(vector))

    return {
        "label": SHARED_LABELS[index],
        "confidence": float(vector[index]),
        "vector": vector,
        "token_count": int(prediction["token_count"]),
        "latency_ms": float(prediction["latency_ms"]),
        "native_label": prediction["label"],
        "task": prediction["task"],
        "probs": {
            label: round(float(vector[position]), 4)
            for position, label in enumerate(SHARED_LABELS)
        },
    }


def text_evidence_payload(
    evidence: TextEvidence | None, now: float
) -> dict[str, Any] | None:
    """Serialise the current text evidence for the client."""

    if evidence is None:
        return None

    return {
        "present": True,
        "label": evidence.label,
        "confidence": round(evidence.confidence, 4),
        "text": evidence.text,
        "token_count": evidence.token_count,
        "age_seconds": round(evidence.age(now), 2),
        "probs": {
            label: round(float(evidence.probabilities[index]), 4)
            for index, label in enumerate(SHARED_LABELS)
        },
    }


# ============================================================
# FRAME PROCESSING
# ============================================================


def decode_frame(payload: str) -> np.ndarray | None:
    """Base64 (with or without a data-URI prefix) -> BGR array."""

    if "," in payload:
        payload = payload.split(",", 1)[1]

    jpeg_bytes = base64.b64decode(payload, validate=True)

    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def process_frame(frame_bgr: np.ndarray, state: SessionState) -> dict[str, Any]:
    """Full pipeline for one frame: face -> fusion -> payload."""

    assert runtime.face is not None and runtime.fusion is not None

    now = time.monotonic()

    elapsed = now - state.previous_frame_time
    state.previous_frame_time = now

    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    face_result = runtime.face.process(frame_bgr, state.face, now=now)

    fused = runtime.fusion.fuse(
        face_probabilities=face_result.smoothed_probabilities,
        face_quality=face_result.quality.overall,
        text_evidence=state.text,
        now=now,
    )

    # Drop text evidence once it has decayed past the cutoff, so a
    # session does not hold a message object alive for its whole life.
    if state.text is not None and not fused.text.present:
        state.text = None

    return {
        "type": "frame",
        "fps": round(fps, 2),
        "face": face_result.as_dict(),
        "text": text_evidence_payload(state.text, now),
        "fusion": fused.as_dict(),
    }


# ============================================================
# WEBSOCKET
# ============================================================


@app.websocket("/ws/video")
async def video_ws(ws: WebSocket) -> None:
    """Carries both modalities for one client session."""

    await ws.accept()

    client = ws.client
    logger.info("WebSocket connected: %s", client)

    state = SessionState(face=runtime.face.session_state())

    # Tell the client what it is actually talking to, so the UI can say
    # "face-only" honestly instead of showing an inert text box.
    await ws.send_json(
        {
            "type": "hello",
            "labels": list(SHARED_LABELS),
            "model": MODEL_KEY,
            "detector": runtime.detector_name,
            "crop_mode": CROP_MODE.value,
            "multicrop": USE_MULTICROP,
            "neutral_margin": NEUTRAL_MARGIN,
            "landmarks": runtime.detector_name != "haar",
            "text_model": runtime.text_model_name,
            "text_task": runtime.text_task,
            "text_available": runtime.text_classifier is not None,
            "tta": USE_TTA,
            "face_provider": runtime.face_provider,
            "text_device": runtime.text_device,
        }
    )

    try:
        while True:
            raw = await ws.receive_text()

            message = _parse_message(raw)

            if message is None:
                await ws.send_json(
                    {"type": "error", "message": "Unrecognised message"}
                )
                continue

            if message["type"] == "text":
                await _handle_text(ws, state, message)
            else:
                await _handle_frame(ws, state, message)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
    except Exception:
        logger.exception("Unexpected error on WebSocket %s", client)
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass


def _parse_message(raw: str) -> dict[str, Any] | None:
    """Accept both the JSON envelope and the original bare-Base64 frame."""

    stripped = raw.lstrip()

    if stripped.startswith("{"):
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if not isinstance(message, dict):
            return None

        kind = message.get("type", "frame")

        if kind not in ("frame", "text"):
            return None

        message["type"] = kind

        return message

    # Legacy protocol: the whole message is the frame.
    return {"type": "frame", "data": raw}


async def _handle_frame(
    ws: WebSocket, state: SessionState, message: dict[str, Any]
) -> None:

    try:
        frame = await run_in_threadpool(decode_frame, message.get("data", ""))
    except Exception as error:
        await ws.send_json({"type": "error", "message": f"Decode error: {error}"})
        return

    if frame is None:
        await ws.send_json({"type": "error", "message": "Failed to decode image"})
        return

    # Off the event loop: ONNX inference and OpenCV both block, and the
    # socket still has to stay responsive to incoming text messages.
    payload = await run_in_threadpool(process_frame, frame, state)

    await ws.send_json(payload)


async def _handle_text(
    ws: WebSocket, state: SessionState, message: dict[str, Any]
) -> None:

    text = str(message.get("text", "")).strip()

    if not text:
        await ws.send_json({"type": "error", "message": "Empty message"})
        return

    if runtime.text_classifier is None:
        await ws.send_json(
            {
                "type": "error",
                "message": "No text model is loaded on the server.",
            }
        )
        return

    result = await run_in_threadpool(classify_text, text[:MAX_TEXT_LENGTH])

    if result is None:
        await ws.send_json({"type": "error", "message": "Text inference failed"})
        return

    # Replace rather than accumulate. The fusion model is "what is the
    # user feeling now", and the newest message is the best evidence
    # for that; older ones are already decaying and keeping a history
    # would double-count a repeated sentiment.
    state.text = TextEvidence(
        probabilities=result["vector"],
        token_count=result["token_count"],
        text=text,
        timestamp=time.monotonic(),
        label=result["label"],
        confidence=result["confidence"],
    )

    await ws.send_json(
        {
            "type": "text_result",
            "id": message.get("id"),
            "text": text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "native_label": result["native_label"],
            "task": result["task"],
            "token_count": result["token_count"],
            "latency_ms": round(result["latency_ms"], 2),
            "probs": result["probs"],
        }
    )


# ============================================================
# HTTP
# ============================================================


class TextRequest(BaseModel):
    text: str


@app.post("/api/text/analyze")
async def analyze_text(request: TextRequest) -> JSONResponse:
    """Stateless one-shot classification.

    Does not touch any session's fusion state, so it is safe to call
    from curl or a test without disturbing a live demo.
    """

    text = request.text.strip()

    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)

    if runtime.text_classifier is None:
        return JSONResponse(
            {"error": "No text model loaded"}, status_code=503
        )

    result = await run_in_threadpool(classify_text, text[:MAX_TEXT_LENGTH])

    return JSONResponse(
        {
            "text": text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "native_label": result["native_label"],
            "task": result["task"],
            "token_count": result["token_count"],
            "latency_ms": round(result["latency_ms"], 2),
            "probs": result["probs"],
        }
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe that also reports which components came up."""

    return {
        "status": "ok",
        "model": MODEL_KEY,
        "device_requested": DEVICE,
        "detector": runtime.detector_name,
        "crop_mode": CROP_MODE.value,
        "multicrop": USE_MULTICROP,
        "neutral_margin": NEUTRAL_MARGIN,
        "landmarks": runtime.detector_name != "haar",
        "tta": USE_TTA,
        "face_provider": runtime.face_provider,
        "text_device": runtime.text_device,
        "gpu": runtime.face_provider != "CPUExecutionProvider"
               or runtime.text_device != "cpu",
        "text_model": runtime.text_model_name,
        "text_task": runtime.text_task,
        "text_available": runtime.text_classifier is not None,
        "labels": list(SHARED_LABELS),
    }


@app.get("/")
async def index() -> FileResponse:
    """Serve the dashboard from the same origin as the WebSocket."""

    return FileResponse(INDEX_PATH)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Serve the multimodal emotion demo.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default=os.environ.get("EMOTION_DEVICE", "auto").lower(),
        help=(
            "Where to run inference. 'auto' (default) uses the GPU when a "
            "provider is available and falls back to CPU; 'gpu' does the "
            "same but fails loudly if no GPU provider registers, which is "
            "what you want when you believe you have one; 'cpu' forces CPU. "
            "Equivalent to EMOTION_DEVICE. On an RTX 4060 the face model "
            "runs ~6.6 ms/frame on CUDA against ~303 ms on CPU."
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=MODEL_KEY,
        help="Which checkpoint to serve. Equivalent to EMOTION_MODEL.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. Off by default - the reloader "
             "re-imports the module, which reloads the ONNX session too.",
    )

    arguments = parser.parse_args()

    # Re-exported so the module-level constants pick them up when uvicorn
    # imports "server:app" in this same process (or a reloaded one).
    os.environ["EMOTION_DEVICE"] = arguments.device
    os.environ["EMOTION_MODEL"] = arguments.model

    uvicorn.run(
        "server:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level="info",
    )
