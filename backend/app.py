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
from starlette.status import WS_1008_POLICY_VIOLATION
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
# Add project root & text-emotion-module to path so imports work regardless
# of how uvicorn is launched.
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "text-emotion-module"))

from models.audio_cnn import AudioCNN                       # noqa: E402
from models.vision_encoder import VisionEncoder             # noqa: E402
from models.text_encoder import TextEncoder                 # noqa: E402
from models.fusion import (                                 # noqa: E402
    MultimodalTransformerFusion,
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_TEXT,
    TARGET_EMOTIONS,
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

# Face detector (MediaPipe, Haar Cascade, or adaptive crop)
_face_detector: Any = None
_detector_type: str = "adaptive_center"

# --- Multimodal Trained Pipeline (data_Finalmodel.pt) ---
_t_fc: nn.Linear | None = None
_audio_cnn_seq: nn.Sequential | None = None
_a_fc: nn.Linear | None = None
_v_fc: nn.Linear | None = None
_fusion_module: MultimodalTransformerFusion | None = None

# --- Text Modality Backbone (BERT) ---
_text_tokenizer: Any = None
_text_model: Any = None

# --- Vision Modality Backbone (ViT) ---
_vision_encoder: VisionEncoder | None = None

# PyTorch device
_device: torch.device = torch.device("cpu")

# Fusion emotion classes (5-class target emotion space)
FUSION_ID2LABEL = {i: name for i, name in enumerate(TARGET_EMOTIONS)}

# ViT-only emotion classes (7-class)
VIT_EMOTION_CLASSES = {}  # populated from emotion_config.json


# ============================================================
# LIFESPAN — load every model once
# ============================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _onnx_session, _onnx_input_name
    global _id2label, _image_size, _mean, _std
    global _face_detector, _detector_type
    global _t_fc, _audio_cnn_seq, _a_fc, _v_fc, _fusion_module
    global _text_tokenizer, _text_model, _vision_encoder
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

    # ---- Face Detection ----
    _face_detector = None
    _detector_type = "adaptive_center"
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            _detector_type = "mediapipe"
            logger.info("MediaPipe solutions FaceDetection initialised")
        else:
            # Check for OpenCV Haar cascade
            cascade_candidates = [
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else "",
                str(_PROJECT_ROOT / "haarcascade_frontalface_default.xml"),
                str(_BACKEND_DIR / "haarcascade_frontalface_default.xml"),
            ]
            for c_path in cascade_candidates:
                if c_path and Path(c_path).exists():
                    cc = cv2.CascadeClassifier(c_path)
                    if not cc.empty():
                        _face_detector = cc
                        _detector_type = "haar"
                        logger.info("OpenCV Haar cascade FaceDetection initialised from %s", c_path)
                        break
            if _face_detector is None:
                logger.info("Using adaptive center-crop face detector (no external detector required)")
    except Exception as exc:
        logger.warning("Face detector init fallback: %s; using adaptive center crop", exc)
        _detector_type = "adaptive_center"

    # ---- Multimodal Pipeline Architecture (matches train_multimodal.py) ----
    _t_fc = nn.Linear(768, 256).to(_device)
    _audio_cnn_seq = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1))
    ).to(_device)
    _a_fc = nn.Linear(128, 256).to(_device)
    _v_fc = nn.Linear(768, 256).to(_device)
    _fusion_module = MultimodalTransformerFusion(embed_dim=256, num_classes=5).to(_device)

    # ---- Load trained weights from data_Finalmodel.pt or best_fusion.pth ----
    final_model_ckpt = _PROJECT_ROOT / "data_Finalmodel.pt"
    if final_model_ckpt.exists():
        try:
            ckpt = torch.load(str(final_model_ckpt), map_location=_device)
            _t_fc.load_state_dict({"weight": ckpt["t_fc.weight"], "bias": ckpt["t_fc.bias"]})
            cnn_dict = {k.replace("audio_cnn.", ""): v for k, v in ckpt.items() if k.startswith("audio_cnn.")}
            _audio_cnn_seq.load_state_dict(cnn_dict)
            _a_fc.load_state_dict({"weight": ckpt["a_fc.weight"], "bias": ckpt["a_fc.bias"]})
            _v_fc.load_state_dict({"weight": ckpt["v_fc.weight"], "bias": ckpt["v_fc.bias"]})
            fusion_dict = {k.replace("fusion.", ""): v for k, v in ckpt.items() if k.startswith("fusion.")}
            _fusion_module.load_state_dict(fusion_dict)
            logger.info("Successfully loaded complete multimodal pipeline from %s", final_model_ckpt)
        except Exception as err:
            logger.error("Failed loading data_Finalmodel.pt: %s", err)
    elif (_PROJECT_ROOT / "best_fusion.pth").exists():
        fusion_ckpt = _PROJECT_ROOT / "best_fusion.pth"
        try:
            _fusion_module.load_state_dict(torch.load(str(fusion_ckpt), map_location=_device))
            logger.info("Fusion weights loaded from %s", fusion_ckpt)
        except Exception as err:
            logger.warning("Could not load best_fusion.pth: %s", err)
    else:
        logger.warning("No multimodal checkpoint found — using random init")

    _t_fc.eval()
    _audio_cnn_seq.eval()
    _a_fc.eval()
    _v_fc.eval()
    _fusion_module.eval()

    # ---- VisionEncoder (ViT [CLS] 768 representation) ----
    _vision_encoder = VisionEncoder(embed_dim=256).to(_device)
    _vision_encoder.eval()
    logger.info("VisionEncoder (ViT 768->256) initialised")

    # ---- BERT Text Backbone (bert-base-uncased) ----
    try:
        from transformers import BertTokenizer, BertModel
        logger.info("Loading BERT text model (bert-base-uncased)...")
        _text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _text_model = BertModel.from_pretrained("bert-base-uncased").to(_device)
        _text_model.eval()
        for p in _text_model.parameters():
            p.requires_grad = False
        logger.info("BERT text model initialised successfully")
    except Exception as exc:
        logger.warning("BERT text model deferred: %s", exc)

    yield  # ── application runs ──

    # teardown
    if _detector_type == "mediapipe" and _face_detector is not None and hasattr(_face_detector, "close"):
        _face_detector.close()
    _face_detector = None
    _onnx_session = None
    _vision_encoder = None
    logger.info("Resources released")


app = FastAPI(
    title="Multimodal Emotion Recognition API",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# TELEMEDICINE CONNECTION MANAGER
# ============================================================

VALID_ROLES = {"patient", "doctor"}

# EMA smoothing factor for temporal prediction stabilisation
EMA_ALPHA = 0.7


class ConnectionManager:
    """Manages role-based WebSocket connections organised by room.

    Each room can have one or more *patient* and *doctor* connections.
    Patients send multimodal data; the resulting prediction is broadcast
    exclusively to doctors in the same room.
    """

    def __init__(self) -> None:
        # room_id -> role -> list[WebSocket]
        self._rooms: dict[str, dict[str, list[WebSocket]]] = {}
        # Task 3: EMA state — room_id -> {emotion: smoothed_prob}
        self._prev_probs: dict[str, dict[str, float]] = {}

    # -- connection lifecycle -------------------------------------------------

    async def connect(self, ws: WebSocket, role: str, room_id: str) -> None:
        await ws.accept()
        room = self._rooms.setdefault(room_id, {"patient": [], "doctor": []})
        room[role].append(ws)
        logger.info(
            "WS connected  role=%s  room=%s  (patients=%d, doctors=%d)",
            role, room_id, len(room["patient"]), len(room["doctor"]),
        )

    def disconnect(self, ws: WebSocket, role: str, room_id: str) -> None:
        room = self._rooms.get(room_id)
        if room and ws in room.get(role, []):
            room[role].remove(ws)
        # clean up empty rooms
        if room and not room["patient"] and not room["doctor"]:
            self._rooms.pop(room_id, None)
            self._prev_probs.pop(room_id, None)
        logger.info("WS disconnected  role=%s  room=%s", role, room_id)

    # -- targeted broadcast ---------------------------------------------------

    async def broadcast_to_doctors(
        self, room_id: str, payload: dict[str, Any]
    ) -> None:
        """Send *payload* to every doctor in *room_id*.

        Applies EMA temporal smoothing to the fusion prediction probabilities
        before broadcasting, and prunes stale / broken connections.
        """
        room = self._rooms.get(room_id)
        if not room:
            return

        # Task 3: EMA smoothing on fusion probabilities
        fusion = payload.get("fusion_prediction")
        if fusion and isinstance(fusion, dict) and fusion.get("probs"):
            current_probs: dict[str, float] = fusion["probs"]
            prev = self._prev_probs.get(room_id, {})

            smoothed: dict[str, float] = {}
            for emotion, cur_val in current_probs.items():
                prev_val = prev.get(emotion, 0.0)
                smoothed[emotion] = EMA_ALPHA * cur_val + (1 - EMA_ALPHA) * prev_val

            # Normalise so probabilities sum to 1.0
            total = sum(smoothed.values())
            if total > 0:
                smoothed = {e: v / total for e, v in smoothed.items()}

            # Update payload with smoothed values
            payload["fusion_prediction"]["probs"] = smoothed

            # Recompute dominant label and confidence from smoothed distribution
            best_emotion = max(smoothed, key=smoothed.get)  # type: ignore[arg-type]
            payload["fusion_prediction"]["label"] = best_emotion
            payload["fusion_prediction"]["confidence"] = round(smoothed[best_emotion], 4)

            # Store for next cycle
            self._prev_probs[room_id] = smoothed

        stale: list[WebSocket] = []
        for doc_ws in room["doctor"]:
            try:
                await doc_ws.send_json(payload)
            except Exception:
                stale.append(doc_ws)
        for ws in stale:
            room["doctor"].remove(ws)
            logger.warning(
                "Pruned stale doctor connection in room=%s", room_id,
            )


_manager = ConnectionManager()


# ============================================================
# SERVE FRONTEND
# ============================================================

@app.get("/")
async def serve_index():
    index_file = _FRONTEND_DIR / "index.html"
    if not index_file.exists():
        index_file = _PROJECT_ROOT / "index.html"
    return FileResponse(str(index_file), media_type="text/html")


@app.get("/call")
async def serve_call_page():
    """Serve the P2P video-call page."""
    call_file = _FRONTEND_DIR / "call.html"
    return FileResponse(str(call_file), media_type="text/html")


@app.get("/doctor")
async def serve_doctor_page():
    """Serve the telemedicine doctor dashboard view."""
    call_file = _FRONTEND_DIR / "call.html"
    return FileResponse(str(call_file), media_type="text/html")


# Serve frontend JS/CSS assets at /static/
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


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
    """Detect face bbox with multi-backend fallback (MediaPipe -> Haar -> Adaptive Center)."""
    h_f, w_f = frame_bgr.shape[:2]

    # 1. MediaPipe
    if _detector_type == "mediapipe" and _face_detector is not None:
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = _face_detector.process(frame_rgb)
            if results and results.detections:
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
        except Exception:
            pass

    # 2. OpenCV Haar Cascade
    if _detector_type == "haar" and _face_detector is not None:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                # Largest face
                best = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
                mx, my = int(w * 0.05), int(h * 0.05)
                bbox = {
                    "x": max(0, x - mx), "y": max(0, y - my),
                    "w": min(w_f, x + w + mx) - max(0, x - mx),
                    "h": min(h_f, y + h + my) - max(0, y - my),
                }
                return bbox, 0.85
        except Exception:
            pass

    # 3. Adaptive Center Crop (telemedicine webcam default)
    # Face is almost always in the center 60% of frame
    box_w = int(w_f * 0.60)
    box_h = int(h_f * 0.65)
    x = max(0, (w_f - box_w) // 2)
    y = max(0, int((h_f - box_h) * 0.40))
    bbox = {
        "x": x,
        "y": y,
        "w": min(w_f - x, box_w),
        "h": min(h_f - y, box_h),
    }
    return bbox, 0.75


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


def extract_vision_feature(face_crop: np.ndarray) -> torch.Tensor | None:
    """Extract 256-D feature representation from face crop using ViT [CLS] (768) -> trained v_fc -> ReLU."""
    if _vision_encoder is None:
        return None
    try:
        tensor_np = preprocess_face(face_crop)
        tensor_torch = torch.from_numpy(tensor_np).to(_device)
        with torch.no_grad():
            if _v_fc is not None:
                cls_768 = _vision_encoder.extract_cls(tensor_torch)
                return torch.relu(_v_fc(cls_768))
            return _vision_encoder(tensor_torch)
    except Exception as exc:
        logger.debug("extract_vision_feature error: %s", exc)
        return None


# ============================================================
# AUDIO HELPERS
# ============================================================

def decode_wav_bytes(raw_b64: str) -> tuple[np.ndarray, int] | None:
    """Decode a base64 WAV or raw PCM float32 chunk → (float32 numpy array, sample_rate)."""
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
            return samples, sr
    except Exception:
        pass

    # Fallback: treat as raw float32 PCM at assumed 22050 Hz
    if len(raw) % 4 == 0 and len(raw) >= 4:
        try:
            samples = np.frombuffer(raw, dtype=np.float32)
            if np.all(np.isfinite(samples)) and len(samples) > 0:
                return samples, 22050
        except Exception:
            pass

    return None


def encode_audio(audio_samples: np.ndarray, actual_sr: int = 22050) -> torch.Tensor | None:
    """audio_samples -> resample to 16 kHz -> MFCC (200, 40) -> audio_cnn_seq -> a_fc -> ReLU -> (1, 256)."""
    if _audio_cnn_seq is None or _a_fc is None:
        return None
    try:
        # Resample to training SR (16000) if browser sends at a different rate
        signal = audio_samples
        if actual_sr != 16000:
            signal = librosa.resample(signal, orig_sr=actual_sr, target_sr=16000)
        # Extract MFCC matching train_multimodal.py (sr=16000, n_mfcc=40, max_len=200)
        mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=40).T
        if len(mfcc) < 200:
            mfcc = np.pad(mfcc, ((0, 200 - len(mfcc)), (0, 0)))
        else:
            mfcc = mfcc[:200]
        mfcc = mfcc.astype(np.float32)

        # (1, 200, 40)
        t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(_device)
        with torch.no_grad():
            a = t.permute(0, 2, 1).unsqueeze(1)  # (1, 1, 40, 200)
            a = _audio_cnn_seq(a)
            a = a.view(a.size(0), -1)
            return torch.relu(_a_fc(a))  # (1, 256)
    except Exception as e:
        logger.debug("encode_audio error: %s", e)
        return None


# ============================================================
# TEXT HELPERS (BERT-base-uncased)
# ============================================================

def get_bert():
    """Lazily load BERT backbone for text sentiment encoding."""
    global _text_tokenizer, _text_model
    if _text_tokenizer is None or _text_model is None:
        try:
            from transformers import BertTokenizer, BertModel
            logger.info("Loading BERT model for real-time text sentiment encoding...")
            _text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            _text_model = BertModel.from_pretrained("bert-base-uncased").to(_device)
            _text_model.eval()
            for p in _text_model.parameters():
                p.requires_grad = False
            logger.info("BERT text model ready")
        except Exception as e:
            logger.warning("Could not load BERT model: %s", e)
    return _text_tokenizer, _text_model


def encode_text(text: str) -> torch.Tensor | None:
    """text -> BERT [CLS] (768) -> trained t_fc -> ReLU -> (1, 256)."""
    if not text or not text.strip() or _t_fc is None:
        return None
    tok, bert_model = get_bert()
    if tok is None or bert_model is None:
        return None

    try:
        tokens = tok(
            text.strip(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            cls_768 = bert_model(**tokens).last_hidden_state[:, 0, :]  # (1, 768)
            return torch.relu(_t_fc(cls_768))  # (1, 256)
    except Exception as exc:
        logger.debug("encode_text error: %s", exc)
        return None


# ============================================================
# FUSION
# ============================================================

def fuse_predictions(
    vision_256: torch.Tensor | None,
    audio_256: torch.Tensor | None,
    text_256: torch.Tensor | None,
) -> dict[str, Any] | None:
    """Run the MultimodalTransformerFusion over available modalities.

    Modality embeddings (0=vision, 1=audio, 2=text) are injected before
    self-attention so the Transformer can distinguish modality sources.

    Returns dict with label, confidence, full 5-class probability distribution,
    and relative modality attention contribution weights.
    """
    if _fusion_module is None:
        return None

    modality_names = []
    if vision_256 is not None:
        modality_names.append("vision")
    if audio_256 is not None:
        modality_names.append("audio")
    if text_256 is not None:
        modality_names.append("text")

    if not modality_names:
        return None

    with torch.no_grad():
        logits, mod_weights = _fusion_module.forward_modalities(
            vision_feat=vision_256,
            audio_feat=audio_256,
            text_feat=text_256,
            return_weights=True,
        )
        if logits is None:
            return None

        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))

        probs_dict = {
            TARGET_EMOTIONS[i]: round(float(probs[i]), 4)
            for i in range(len(TARGET_EMOTIONS))
        }

        return {
            "label": TARGET_EMOTIONS[idx],
            "confidence": round(float(probs[idx]), 4),
            "probs": probs_dict,
            "modalities_used": modality_names,
            "modality_weights": mod_weights,
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
                        vit_pred = None
                        if _onnx_session is not None:
                            vit_pred, _ = run_vit_inference(face_crop)
                        vision_256 = extract_vision_feature(face_crop)

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
            wav_result = decode_wav_bytes(audio_b64)
            if wav_result is not None:
                samples, actual_sr = wav_result
                if len(samples) > 1024:
                    audio_256 = encode_audio(samples, actual_sr)
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
        print(f"Vision: {vision_256 is not None}, Audio: {audio_256 is not None}, Text: {text_256 is not None}")
        fusion_result = fuse_predictions(vision_256, audio_256, text_256)
        payload["fusion_prediction"] = fusion_result
    except Exception as e:
        logger.debug("Fusion error: %s", e)

    return payload, current_time


# ============================================================
# TELEMEDICINE WEBSOCKET ENDPOINT (role-based rooms)
# ============================================================

@app.websocket("/ws/multimodal/{role}/{room_id}")
async def multimodal_room_ws(ws: WebSocket, role: str, room_id: str) -> None:
    """Role-aware multimodal WebSocket for telemedicine rooms.

    Patients → send image / audio / text payloads.
              Inference is offloaded to the thread pool.
              Results are broadcast to doctors in the same room.

    Doctors  → receive-only; predictions arrive as JSON.
    """
    if role not in VALID_ROLES:
        await ws.close(code=WS_1008_POLICY_VIOLATION, reason=f"Invalid role '{role}'")
        return

    await _manager.connect(ws, role, room_id)
    prev_time = time.time()
    loop = asyncio.get_running_loop()

    try:
        while True:
            raw: str = await ws.receive_text()

            if role == "doctor":
                # Doctors are receive-only; ignore any data they send.
                continue

            # --- patient path: parse, infer, broadcast to doctors ---
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"image": raw}

            image_b64 = msg.get("image")
            audio_b64 = msg.get("audio")
            text = msg.get("text")

            payload, prev_time = await loop.run_in_executor(
                _executor,
                process_multimodal,
                image_b64,
                audio_b64,
                text,
                prev_time,
            )

            # Enrich payload with room metadata
            payload["room_id"] = room_id
            payload["source_role"] = "patient"

            # Broadcast to doctors only — never echo prediction back to the patient
            await _manager.broadcast_to_doctors(room_id, payload)

            # Acknowledge receipt to the patient to release backpressure
            await ws.send_json({"status": "acknowledged"})

    except WebSocketDisconnect:
        logger.info("Room WS disconnected: role=%s room=%s", role, room_id)
    except Exception:
        logger.exception(
            "Unexpected error on room WS role=%s room=%s", role, room_id,
        )
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass
    finally:
        _manager.disconnect(ws, role, room_id)


# ============================================================
# WEBSOCKET ENDPOINT (original — no rooms)
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
# WEBRTC SIGNALING ENDPOINT
# ============================================================

# Room → list of connected signaling WebSockets (max 2 per room)
_signal_rooms: dict[str, list[WebSocket]] = {}


@app.websocket("/ws/signal/{room_id}")
async def signaling_ws(ws: WebSocket, room_id: str) -> None:
    """Lightweight signaling relay for WebRTC peer connections.

    Two browsers connect to the same room_id.  The first one becomes
    the *initiator* (sends the offer); the second becomes the *joiner*
    (sends the answer).  All {"type": "signal", "data": ...}
    messages are forwarded to the other peer in the room.

    This endpoint carries **no ML logic** — it only relays SDP and
    ICE blobs so the two browsers can establish a direct P2P link.
    """
    await ws.accept()

    room = _signal_rooms.setdefault(room_id, [])

    if len(room) >= 2:
        await ws.close(code=WS_1008_POLICY_VIOLATION, reason="Room is full (max 2 peers)")
        return

    room.append(ws)
    is_initiator = len(room) == 1
    logger.info("Signal WS joined room=%s (peer #%d)", room_id, len(room))

    # Tell this peer its role
    await ws.send_json({"type": "ready", "initiator": is_initiator})

    # If a second peer just joined, also tell the first peer to start
    if len(room) == 2:
        try:
            await room[0].send_json({"type": "ready", "initiator": True})
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Relay signal data to the OTHER peer in the room
            for other_ws in room:
                if other_ws is not ws:
                    try:
                        await other_ws.send_json(msg)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info("Signal WS left room=%s", room_id)
    except Exception:
        logger.exception("Signal WS error room=%s", room_id)
    finally:
        if ws in room:
            room.remove(ws)
        # Notify remaining peer that the other side left
        for other_ws in room:
            try:
                await other_ws.send_json({"type": "peer_left"})
            except Exception:
                pass
        # Clean up empty rooms
        if not room:
            _signal_rooms.pop(room_id, None)


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

    # Support running either from root (uvicorn backend.app:app) or within backend directory (uvicorn app:app)
    app_target = "backend.app:app" if Path.cwd() == _PROJECT_ROOT else "app:app"
    uvicorn.run(
        app_target,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
