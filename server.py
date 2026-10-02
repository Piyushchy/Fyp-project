"""
Multimodal Emotion Recognition — FastAPI WebSocket Server
==========================================================

Accepts JSON payloads over WebSocket containing:
  • base64 JPEG frame  (video)
  • base64 WAV chunk   (audio)
  • transcribed text   (text / speech-to-text)

Routes each modality through its own encoder to produce a 256-dim
feature vector, then fuses all available vectors through the
TransformerFusion module for a final emotion prediction.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import struct
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import librosa
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
# Add project root & text-emotion-module to path so imports work regardless
# of how uvicorn is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "text-emotion-module"))

from audio_cnn import AudioCNN                       # noqa: E402
from main import (                                    # noqa: E402
    MultimodalTransformerFusion,
    TextEncoder,
)

# Text module
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: E402

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

_VIT_DIR = _PROJECT_ROOT / "vit-emotion-v2-deployment"
ONNX_PATH = _VIT_DIR / "vit-emotion-v2.onnx"
CONFIG_PATH = _VIT_DIR / "emotion_config.json"
TEXT_CHECKPOINT_DIR = _PROJECT_ROOT / "text-emotion-module" / "checkpoints" / "tinybert"

# ============================================================
# THREAD POOL — keeps inference off the event loop
# ============================================================

_executor = ThreadPoolExecutor(max_workers=2)

# ============================================================
# GLOBAL SINGLETONS (populated at startup)
# ============================================================

# --- Vision (ONNX ViT) ---
_onnx_session: ort.InferenceSession | None = None
_onnx_input_name: str = ""
_id2label: dict[int, str] = {}
_image_size: int = 224
_mean: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_std: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# MediaPipe face detector
_mp_face_detection: mp.solutions.face_detection.FaceDetection | None = None

# --- Audio (AudioCNN encoder) ---
_audio_cnn: AudioCNN | None = None

# --- Text (TinyBERT) ---
_text_tokenizer = None
_text_model = None

# --- Fusion ---
_fusion_module: MultimodalTransformerFusion | None = None
_text_encoder: TextEncoder | None = None

# PyTorch device
_device: torch.device = torch.device("cpu")

# Fusion emotion classes (5-class from main.py)
FUSION_ID2LABEL = {0: "happy", 1: "sad", 2: "angry", 3: "fear", 4: "surprise"}

# ViT-only emotion classes (7-class)
VIT_EMOTION_CLASSES = {}  # populated from emotion_config.json


# ============================================================
# LIFESPAN — load every model once
# ============================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _onnx_session, _onnx_input_name
    global _id2label, _image_size, _mean, _std
    global _mp_face_detection
    global _audio_cnn
    global _text_tokenizer, _text_model
    global _fusion_module, _text_encoder
    global _device, VIT_EMOTION_CLASSES

    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("PyTorch device: %s", _device)

    # ---- ViT ONNX config ----
    with open(CONFIG_PATH, "r") as fh:
        config = json.load(fh)
    _id2label = {int(k): v for k, v in config["id2label"].items()}
    VIT_EMOTION_CLASSES = _id2label
    _image_size = int(config.get("image_size", 224))
    _mean = np.array(config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    _std = np.array(config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)
    logger.info("ViT emotion classes: %s", _id2label)

    # ---- ViT ONNX session ----
    logger.info("Loading ONNX model from %s …", ONNX_PATH)
    _onnx_session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    _onnx_input_name = _onnx_session.get_inputs()[0].name
    logger.info("ONNX model loaded — input: %s", _onnx_input_name)

    # ---- MediaPipe ----
    _mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    logger.info("MediaPipe FaceDetection initialised")

    # ---- AudioCNN encoder ----
    _audio_cnn = AudioCNN(num_classes=5).to(_device)
    audio_ckpt = _PROJECT_ROOT / "best_audio_cnn.pth"
    if audio_ckpt.exists():
        _audio_cnn.load_state_dict(torch.load(str(audio_ckpt), map_location=_device))
        logger.info("AudioCNN weights loaded from %s", audio_ckpt)
    else:
        logger.warning("No AudioCNN checkpoint at %s — using random init", audio_ckpt)
    _audio_cnn.eval()

    # ---- Text model (TinyBERT) ----
    if TEXT_CHECKPOINT_DIR.exists():
        _text_tokenizer = AutoTokenizer.from_pretrained(str(TEXT_CHECKPOINT_DIR))
        _text_model = AutoModelForSequenceClassification.from_pretrained(
            str(TEXT_CHECKPOINT_DIR)
        ).to(_device)
        _text_model.eval()
        logger.info("TinyBERT text model loaded from %s", TEXT_CHECKPOINT_DIR)
    else:
        logger.warning("No TinyBERT checkpoint at %s — text disabled", TEXT_CHECKPOINT_DIR)

    # ---- TextEncoder (768→256 projection) ----
    _text_encoder = TextEncoder(embed_dim=256).to(_device)
    _text_encoder.eval()
    logger.info("TextEncoder projection layer initialised")

    # ---- Fusion module ----
    _fusion_module = MultimodalTransformerFusion(
        embed_dim=256, num_classes=5
    ).to(_device)
    fusion_ckpt = _PROJECT_ROOT / "best_fusion.pth"
    if fusion_ckpt.exists():
        _fusion_module.load_state_dict(
            torch.load(str(fusion_ckpt), map_location=_device)
        )
        logger.info("Fusion weights loaded from %s", fusion_ckpt)
    else:
        logger.warning("No fusion checkpoint at %s — using random init", fusion_ckpt)
    _fusion_module.eval()

    yield  # ── application runs ──

    # teardown
    if _mp_face_detection is not None:
        _mp_face_detection.close()
    _onnx_session = None
    logger.info("Resources released")


app = FastAPI(
    title="Multimodal Emotion Recognition API",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# SERVE FRONTEND
# ============================================================

@app.get("/")
async def serve_index():
    return FileResponse(str(_PROJECT_ROOT / "index.html"), media_type="text/html")


# ============================================================
# VISION HELPERS
# ============================================================

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalise, reshape a BGR face crop to (1,3,224,224)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (_image_size, _image_size))
    image = face_resized.astype(np.float32) / 255.0
    image = (image - _mean) / _std
    image = image.transpose(2, 0, 1)          # HWC → CHW
    return np.expand_dims(image, 0).astype(np.float32)  # NCHW


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def detect_face(frame_bgr: np.ndarray):
    """MediaPipe face detection → best bbox with 5% margin."""
    assert _mp_face_detection is not None
    h_f, w_f = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_detection.process(frame_rgb)
    if not results.detections:
        return None, None
    best = max(results.detections, key=lambda d: d.score[0])
    confidence = float(best.score[0])
    rbb = best.location_data.relative_bounding_box
    x, y = int(rbb.xmin * w_f), int(rbb.ymin * h_f)
    w, h = int(rbb.width * w_f), int(rbb.height * h_f)
    mx, my = int(w * 0.05), int(h * 0.05)
    bbox = {
        "x": max(0, x - mx), "y": max(0, y - my),
        "w": min(w_f, x + w + mx) - max(0, x - mx),
        "h": min(h_f, y + h + my) - max(0, y - my),
    }
    return bbox, confidence


def run_vit_inference(face_crop: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Run ViT ONNX → return (label+conf dict, raw logit vector)."""
    assert _onnx_session is not None
    tensor = preprocess_face(face_crop)
    logits = _onnx_session.run(None, {_onnx_input_name: tensor})[0][0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {
        "label": _id2label.get(idx, f"class_{idx}"),
        "confidence": round(float(probs[idx]), 4),
    }, logits


def vit_logits_to_256(logits: np.ndarray) -> torch.Tensor:
    """Project the ViT's 7-dim logit vector into a 256-dim feature via a
    learnable linear layer stored as a module attribute on this function.

    On first call the layer is lazily created and moved to _device.
    """
    if not hasattr(vit_logits_to_256, "_proj"):
        proj = nn.Linear(len(_id2label), 256).to(_device)
        proj.eval()
        # Try loading saved projection weights
        proj_ckpt = _PROJECT_ROOT / "vit_proj_256.pth"
        if proj_ckpt.exists():
            proj.load_state_dict(torch.load(str(proj_ckpt), map_location=_device))
            logger.info("ViT→256 projection loaded from %s", proj_ckpt)
        else:
            logger.warning("No vit_proj_256.pth — using random projection")
        vit_logits_to_256._proj = proj

    with torch.no_grad():
        t = torch.tensor(logits, dtype=torch.float32).unsqueeze(0).to(_device)
        return vit_logits_to_256._proj(t)  # (1, 256)


# ============================================================
# AUDIO HELPERS
# ============================================================

def decode_wav_bytes(raw_b64: str) -> np.ndarray | None:
    """Decode a base64 WAV or raw PCM float32 chunk → float32 numpy array."""
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except Exception:
        return None

    # Try WAV container first
    try:
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            pcm = wf.readframes(n_frames)
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            # Convert to float32
            if sampwidth == 2:
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples[::n_channels]  # take first channel
            return samples
    except Exception:
        pass

    # Fallback: treat as raw float32 PCM
    if len(raw) % 4 == 0 and len(raw) >= 4:
        try:
            samples = np.frombuffer(raw, dtype=np.float32)
            if np.all(np.isfinite(samples)) and len(samples) > 0:
                return samples
        except Exception:
            pass

    return None


def extract_mfcc_from_array(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    hop_length: int = 512,
    max_len: int = 130,
) -> np.ndarray:
    """Extract normalised MFCCs from a float32 waveform array.

    Returns shape (40, max_len).
    """
    # Resample if needed — browser typically sends 44100 or 48000
    # librosa.feature.mfcc handles the sr parameter internally.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Pad or truncate
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # Per-feature normalisation
    m = np.mean(mfcc, axis=1, keepdims=True)
    s = np.std(mfcc, axis=1, keepdims=True) + 1e-8
    return ((mfcc - m) / s).astype(np.float32)


def encode_audio(audio_samples: np.ndarray) -> torch.Tensor:
    """audio_samples (float32 1-D) → AudioCNN encoder → (1, 256)."""
    assert _audio_cnn is not None
    mfcc = extract_mfcc_from_array(audio_samples)
    # (1, 1, 40, 130)
    t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        return _audio_cnn.forward_encoder(t)  # (1, 256)


# ============================================================
# TEXT HELPERS
# ============================================================

def encode_text(text: str) -> torch.Tensor | None:
    """text → TinyBERT hidden state (768) → TextEncoder → (1, 256)."""
    if _text_model is None or _text_tokenizer is None or _text_encoder is None:
        return None
    if not text or not text.strip():
        return None

    encoded = _text_tokenizer(
        text.strip(),
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _text_model(**encoded, output_hidden_states=True)
        # Use [CLS] token from last hidden state → (1, hidden_size)
        cls_hidden = outputs.hidden_states[-1][:, 0, :]  # (1, hidden_size)

        # TinyBERT hidden size is 312, not 768 — handle dynamically
        hidden_dim = cls_hidden.shape[1]

        # Lazy-init a projection that matches actual hidden dim
        if not hasattr(encode_text, "_proj") or encode_text._proj_dim != hidden_dim:
            encode_text._proj = nn.Linear(hidden_dim, 256).to(_device)
            encode_text._proj.eval()
            encode_text._proj_dim = hidden_dim

            txt_proj_ckpt = _PROJECT_ROOT / "text_proj_256.pth"
            if txt_proj_ckpt.exists():
                encode_text._proj.load_state_dict(
                    torch.load(str(txt_proj_ckpt), map_location=_device)
                )
                logger.info("Text→256 projection loaded from %s", txt_proj_ckpt)
            else:
                logger.warning("No text_proj_256.pth — using random projection")

        return encode_text._proj(cls_hidden)  # (1, 256)


# ============================================================
# FUSION
# ============================================================

def fuse_predictions(
    vision_256: torch.Tensor | None,
    audio_256: torch.Tensor | None,
    text_256: torch.Tensor | None,
) -> dict[str, Any] | None:
    """Run the TransformerFusion over available modalities.

    The fusion module expects (audio, text) as two sequence tokens.
    We extend it to handle 1–3 tokens by stacking whatever is available
    and mean-pooling inside the transformer.

    Returns dict with label + confidence, or None if < 1 modality present.
    """
    assert _fusion_module is not None

    vectors = []
    modality_names = []
    if vision_256 is not None:
        vectors.append(vision_256)
        modality_names.append("vision")
    if audio_256 is not None:
        vectors.append(audio_256)
        modality_names.append("audio")
    if text_256 is not None:
        vectors.append(text_256)
        modality_names.append("text")

    if len(vectors) == 0:
        return None

    with torch.no_grad():
        # Stack → (1, N, 256)
        seq = torch.cat([v.unsqueeze(1) if v.dim() == 2 else v.unsqueeze(0).unsqueeze(0) for v in vectors], dim=1)

        # Transformer self-attention across modality tokens
        transformer_out = _fusion_module.transformer(seq)  # (1, N, 256)
        pooled = torch.mean(transformer_out, dim=1)        # (1, 256)

        # Classifier — BatchNorm1d needs eval mode (which we set at startup)
        logits = _fusion_module.classifier(pooled)          # (1, 5)
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())

        return {
            "label": FUSION_ID2LABEL.get(idx, f"class_{idx}"),
            "confidence": round(float(probs[idx]), 4),
            "modalities_used": modality_names,
        }


# ============================================================
# FULL FRAME PROCESSOR (runs in thread pool)
# ============================================================

def process_multimodal(
    image_b64: str | None,
    audio_b64: str | None,
    text: str | None,
    prev_time: float,
) -> tuple[dict[str, Any], float]:
    """Orchestrate all three modality encoders + fusion.

    This function is CPU/GPU-bound and should be called from a thread pool.
    """
    current_time = time.time()
    elapsed = current_time - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    payload: dict[str, Any] = {
        "fps": round(fps, 2),
        "face_detected": False,
        "bbox": None,
        "vit_prediction": None,
        "mediapipe_confidence": None,
        "fusion_prediction": None,
        "transcript": text or "",
    }

    # ── 1. VISION ──
    vision_256: torch.Tensor | None = None
    if image_b64:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            jpeg_bytes = base64.b64decode(image_b64, validate=True)
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is not None:
                bbox, mp_conf = detect_face(frame_bgr)
                if bbox is not None:
                    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                    face_crop = frame_bgr[y : y + h, x : x + w]
                    if face_crop.size > 0:
                        vit_pred, vit_logits = run_vit_inference(face_crop)
                        vision_256 = vit_logits_to_256(vit_logits)

                        payload["face_detected"] = True
                        payload["bbox"] = bbox
                        payload["vit_prediction"] = vit_pred
                        payload["mediapipe_confidence"] = (
                            round(mp_conf, 4) if mp_conf else None
                        )
        except Exception as e:
            logger.debug("Vision processing error: %s", e)

    # ── 2. AUDIO ──
    audio_256: torch.Tensor | None = None
    if audio_b64:
        try:
            samples = decode_wav_bytes(audio_b64)
            if samples is not None and len(samples) > 1024:
                audio_256 = encode_audio(samples)
        except Exception as e:
            logger.debug("Audio processing error: %s", e)

    # ── 3. TEXT ──
    text_256: torch.Tensor | None = None
    if text and text.strip():
        try:
            text_256 = encode_text(text)
        except Exception as e:
            logger.debug("Text processing error: %s", e)

    # ── 4. FUSION ──
    try:
        fusion_result = fuse_predictions(vision_256, audio_256, text_256)
        payload["fusion_prediction"] = fusion_result
    except Exception as e:
        logger.debug("Fusion error: %s", e)

    return payload, current_time


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/multimodal")
async def multimodal_ws(ws: WebSocket) -> None:
    """
    Accept multimodal JSON payloads over WebSocket.

    Client → Server (JSON):
        {
            "image": "<base64 JPEG>",
            "audio": "<base64 WAV/PCM>",
            "text":  "transcribed speech"
        }

    Server → Client (JSON):
        Full prediction payload including per-modality + fusion results.
    """
    await ws.accept()
    client = ws.client
    logger.info("Multimodal WebSocket connected: %s", client)

    prev_time = time.time()
    loop = asyncio.get_running_loop()

    try:
        while True:
            raw: str = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                # Backward compat: treat as plain base64 image
                msg = {"image": raw}

            image_b64 = msg.get("image")
            audio_b64 = msg.get("audio")
            text = msg.get("text")

            # Run blocking inference in thread pool
            payload, prev_time = await loop.run_in_executor(
                _executor,
                process_multimodal,
                image_b64,
                audio_b64,
                text,
                prev_time,
            )

            await ws.send_json(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
    except Exception:
        logger.exception("Unexpected error on WebSocket %s", client)
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass


# Keep the old vision-only endpoint for backward compatibility
@app.websocket("/ws/video")
async def video_ws(ws: WebSocket) -> None:
    """Legacy vision-only WebSocket — wraps multimodal handler."""
    await ws.accept()
    client = ws.client
    logger.info("Legacy video WebSocket connected: %s", client)

    prev_time = time.time()
    loop = asyncio.get_running_loop()

    try:
        while True:
            data: str = await ws.receive_text()

            payload, prev_time = await loop.run_in_executor(
                _executor,
                process_multimodal,
                data,   # image
                None,   # no audio
                None,   # no text
                prev_time,
            )
            await ws.send_json(payload)

    except WebSocketDisconnect:
        logger.info("Legacy WebSocket disconnected: %s", client)
    except Exception:
        logger.exception("Unexpected error on legacy WebSocket %s", client)
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ============================================================
# ENTRYPOINT
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
