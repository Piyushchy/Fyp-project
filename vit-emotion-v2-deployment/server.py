"""
FastAPI WebSocket backend for real-time facial emotion recognition.

Accepts Base64-encoded JPEG frames over WebSocket, runs MediaPipe face
detection → ViT ONNX inference, and returns per-frame JSON predictions.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# PATHS  (relative to this file)
# ============================================================

_BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = _BASE_DIR / "vit-emotion-v2.onnx"
CONFIG_PATH = _BASE_DIR / "emotion_config.json"

# ============================================================
# GLOBAL SINGLETONS  (populated on startup, torn down on shutdown)
# ============================================================

_onnx_session: ort.InferenceSession | None = None
_onnx_input_name: str = ""
_id2label: dict[int, str] = {}
_image_size: int = 224
_mean: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_std: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# MediaPipe face detector — reused across frames / connections.
_mp_face_detection: mp.solutions.face_detection.FaceDetection | None = None


# ============================================================
# LIFESPAN  (startup / shutdown)
# ============================================================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load heavy resources once at startup; release on shutdown."""
    global _onnx_session, _onnx_input_name
    global _id2label, _image_size, _mean, _std
    global _mp_face_detection

    # --- config ---
    with open(CONFIG_PATH, "r") as fh:
        config = json.load(fh)

    _id2label = {int(k): v for k, v in config["id2label"].items()}
    _image_size = int(config.get("image_size", 224))
    _mean = np.array(config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    _std = np.array(config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)

    logger.info("Emotion classes: %s", _id2label)

    # --- ONNX ---
    logger.info("Loading ONNX model from %s …", ONNX_PATH)
    _onnx_session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    _onnx_input_name = _onnx_session.get_inputs()[0].name
    logger.info("ONNX model loaded — input name: %s", _onnx_input_name)

    # --- MediaPipe ---
    _mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0,       # 0 = short-range (< 2 m), best for webcam
        min_detection_confidence=0.5,
    )
    logger.info("MediaPipe FaceDetection initialised")

    yield  # ── application runs ──

    # --- teardown ---
    if _mp_face_detection is not None:
        _mp_face_detection.close()
    _onnx_session = None
    logger.info("Resources released")


app = FastAPI(
    title="Emotion Recognition WebSocket API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# PREPROCESSING  (mirrors realtime_inference_onnx.py)
# ============================================================


def preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalise, and reshape a BGR face crop to (1, 3, 224, 224)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (_image_size, _image_size))

    image = face_resized.astype(np.float32) / 255.0
    image = (image - _mean) / _std
    image = image.transpose(2, 0, 1)             # HWC → CHW
    image = np.expand_dims(image, axis=0)         # CHW → NCHW

    return image.astype(np.float32)


# ============================================================
# SOFTMAX  (mirrors realtime_inference_onnx.py)
# ============================================================


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1-D logit vector."""
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


# ============================================================
# FACE DETECTION  (MediaPipe)
# ============================================================


def detect_face(
    frame_bgr: np.ndarray,
) -> tuple[dict[str, int] | None, float | None]:
    """
    Run MediaPipe face detection and return the largest face bounding box
    with a 5 % margin, plus the detector's confidence score.

    Returns
    -------
    bbox : dict | None
        ``{"x": int, "y": int, "w": int, "h": int}`` in pixel coords, or
        ``None`` when no face is found.
    confidence : float | None
        MediaPipe detection confidence, or ``None``.
    """
    assert _mp_face_detection is not None

    h_frame, w_frame = frame_bgr.shape[:2]

    # MediaPipe expects RGB input.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_detection.process(frame_rgb)

    if not results.detections:
        return None, None

    # Pick the detection with the highest score.
    best = max(results.detections, key=lambda d: d.score[0])
    confidence = float(best.score[0])

    rbb = best.location_data.relative_bounding_box
    x = int(rbb.xmin * w_frame)
    y = int(rbb.ymin * h_frame)
    w = int(rbb.width * w_frame)
    h = int(rbb.height * h_frame)

    # Apply a 5 % margin around the detected box.
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_frame, x + w + margin_x)
    y2 = min(h_frame, y + h + margin_y)

    bbox = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    return bbox, confidence


# ============================================================
# INFERENCE
# ============================================================


def run_inference(face_crop: np.ndarray) -> dict[str, Any]:
    """
    Preprocess a BGR face crop, run ONNX inference, and return the
    predicted label + confidence.
    """
    assert _onnx_session is not None

    tensor = preprocess(face_crop)
    logits = _onnx_session.run(None, {_onnx_input_name: tensor})[0][0]

    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    return {
        "label": _id2label.get(pred_idx, f"class_{pred_idx}"),
        "confidence": round(conf, 4),
    }


# ============================================================
# FRAME PROCESSING  (single-frame orchestrator)
# ============================================================


def process_frame(
    frame_bgr: np.ndarray,
    prev_time: float,
) -> tuple[dict[str, Any], float]:
    """
    Full pipeline for one frame:
      decode → detect → crop → infer → build payload.

    Returns the JSON-serialisable payload dict and the current timestamp
    (for FPS tracking in the next call).
    """
    current_time = time.time()
    elapsed = current_time - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    bbox, mp_conf = detect_face(frame_bgr)

    if bbox is None:
        payload: dict[str, Any] = {
            "face_detected": False,
            "bbox": None,
            "vit_prediction": None,
            "mediapipe_confidence": None,
            "fps": round(fps, 2),
        }
        return payload, current_time

    # Crop the face region from the frame.
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    face_crop = frame_bgr[y : y + h, x : x + w]

    # Guard against degenerate crops.
    if face_crop.size == 0:
        payload = {
            "face_detected": False,
            "bbox": None,
            "vit_prediction": None,
            "mediapipe_confidence": round(mp_conf, 4) if mp_conf else None,
            "fps": round(fps, 2),
        }
        return payload, current_time

    prediction = run_inference(face_crop)

    payload = {
        "face_detected": True,
        "bbox": bbox,
        "vit_prediction": prediction,
        "mediapipe_confidence": round(mp_conf, 4) if mp_conf else None,
        "fps": round(fps, 2),
    }
    return payload, current_time


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================


@app.websocket("/ws/video")
async def video_ws(ws: WebSocket) -> None:
    """
    Accept a WebSocket connection that streams Base64-encoded JPEG frames.

    Protocol
    --------
    * **Client → Server** : plain-text message containing the raw Base64
      string of a JPEG-encoded image (no data-URI prefix).
    * **Server → Client** : JSON string with emotion prediction results.
    """
    await ws.accept()
    client = ws.client
    logger.info("WebSocket connected: %s", client)

    prev_time = time.time()

    try:
        while True:
            # Receive a Base64-encoded JPEG frame.
            data: str = await ws.receive_text()

            # --- decode Base64 → JPEG bytes → OpenCV BGR array ---
            try:
                # Strip optional data-URI prefix for robustness.
                if "," in data:
                    data = data.split(",", 1)[1]

                jpeg_bytes = base64.b64decode(data, validate=True)
                np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame_bgr is None:
                    await ws.send_json({"error": "Failed to decode image"})
                    continue
            except Exception as decode_err:
                await ws.send_json({"error": f"Decode error: {decode_err}"})
                continue

            # --- run full pipeline ---
            payload, prev_time = process_frame(frame_bgr, prev_time)
            await ws.send_json(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
    except Exception:
        logger.exception("Unexpected error on WebSocket %s", client)
        # Attempt a graceful close; ignore failures if already closed.
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


# ============================================================
# ENTRYPOINT  (for `python server.py`)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
