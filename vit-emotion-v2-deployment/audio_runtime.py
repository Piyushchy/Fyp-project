# ============================================================
# AUDIO RUNTIME
# ============================================================
#
# Serving-side wrapper around the trained WavLM encoder in
# ../audio-emotion-module. Turns a stream of microphone samples
# into the AudioEvidence that fusion.py pools.
#
# The contract this file has to honour
# ------------------------------------
# A model is only as good as the match between what it was
# trained on and what it is served. prepare_audio_data.py put
# every training clip through a specific sequence - 16 kHz mono,
# silence trimmed at 30 dB, the most energetic 4-second window,
# peak-normalised to 0.95 - and a live window that skips any of
# those steps is out of distribution. Peak normalisation matters
# most: the corpora differ by over 20 dB in level, the model was
# explicitly denied that cue, and a laptop mic is quieter than
# all of them. Feeding raw microphone gain would hand it an
# input unlike anything it saw.
#
# So the preprocessing here is a deliberate duplicate of the
# training path, not a reimplementation of it, and the constants
# are imported from the audio module rather than copied.
#
# Why the import dance
# --------------------
# The audio and text modules both ship config.py, data.py and
# modeling.py, and server.py already puts the text module on
# sys.path. Python's top-level namespace is flat, so a plain
# `import config` here would get whichever module won the race.
# The names are parked for the duration of the load. Same
# problem and same fix as audio-emotion-module/eval_multimodal.py.
# ============================================================

from __future__ import annotations

import contextlib
import logging
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_MODULE_DIR = BASE_DIR.parent / "audio-emotion-module"

COLLIDING_MODULES = (
    "config", "data", "metrics", "modeling", "calibration", "predict",
    "audio_cnn",
)


@contextlib.contextmanager
def _audio_namespace():
    """Let the audio module's files import each other, then undo it."""

    parked = {
        name: sys.modules.pop(name)
        for name in COLLIDING_MODULES
        if name in sys.modules
    }

    saved_path = list(sys.path)

    # The MFCC baseline imports audio_cnn from the repository root.
    sys.path.insert(0, str(AUDIO_MODULE_DIR))
    sys.path.insert(1, str(BASE_DIR.parent))

    try:
        yield
    finally:
        sys.path[:] = saved_path

        for name in COLLIDING_MODULES:
            sys.modules.pop(name, None)

        sys.modules.update(parked)


# ============================================================
# ROLLING BUFFER
# ============================================================


class AudioBuffer:
    """The last few seconds of one session's microphone, at 16 kHz.

    A deque of chunks rather than one growing array: the browser
    sends small blocks continuously, and repeatedly concatenating
    into a single buffer is quadratic over a long session.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0
        self._lock = threading.Lock()

    def extend(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)

        if samples.size == 0:
            return

        with self._lock:
            self._chunks.append(samples)
            self._size += samples.size

            while self._size - self._chunks[0].size >= self.capacity:
                self._size -= self._chunks.popleft().size

    @property
    def seconds_held(self) -> float:
        with self._lock:
            return self._size / 16000.0

    def window(self, length: int) -> np.ndarray | None:
        """The most recent `length` samples, or None if short."""

        with self._lock:
            if self._size < length:
                return None

            joined = np.concatenate(list(self._chunks))

        return joined[-length:]


# ============================================================
# THE MODEL
# ============================================================


class AudioEmotionRuntime:
    """Trained WavLM encoder plus the training-time preprocessing."""

    def __init__(self, model_key: str = "wavlm-base-plus", device: str = "auto"):
        import torch

        with _audio_namespace():
            import config as audio_config
            import modeling as audio_modeling

            self.sample_rate = int(audio_config.SAMPLE_RATE)
            self.clip_samples = int(audio_config.CLIP_SAMPLES)
            self.silence_top_db = float(audio_config.SILENCE_TOP_DB)
            self.min_voiced_seconds = float(audio_config.MIN_VOICED_SECONDS)
            self.min_rms = float(audio_config.MIN_RMS)
            self.labels = list(audio_config.LABELS)

            checkpoint = audio_config.checkpoint_path(model_key) / "model.pt"

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"No audio checkpoint at {checkpoint}. Train it with:\n"
                    f"  cd audio-emotion-module && "
                    f"python train_audio.py --model {model_key}"
                )

            self.display_name = audio_config.MODELS[model_key]["display_name"]

            model = audio_modeling.build(model_key)

        if device == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        state = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(state)

        self.model = model.to(self.device).eval()
        self.model_key = model_key
        self._torch = torch

        # The encoder is not thread-safe to call concurrently from the
        # thread-pool workers the server dispatches frames on.
        self._lock = threading.Lock()

    # --------------------------------------------------------

    def condition(self, samples: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """Mirror prepare_audio_data.condition for one live window.

        Returns (clip, voiced_seconds, rms) or None when the window
        holds nothing usable.
        """

        import librosa

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)

        if samples.size == 0:
            return None

        trimmed, _ = librosa.effects.trim(samples, top_db=self.silence_top_db)

        if trimmed.size == 0:
            return None

        voiced_seconds = trimmed.size / self.sample_rate

        if trimmed.size <= self.clip_samples:
            clip = np.zeros(self.clip_samples, dtype=np.float32)
            clip[: trimmed.size] = trimmed
        else:
            clip = trimmed[-self.clip_samples :]

        rms = float(np.sqrt(np.mean(np.square(clip))))

        # Peak-normalise, exactly as training did. Without this the
        # model is served a systematically quieter input than anything
        # in its corpus.
        peak = float(np.abs(clip).max())

        if peak > 0:
            clip = clip / peak * 0.95

        return clip, voiced_seconds, rms

    def predict(self, samples: np.ndarray) -> tuple[np.ndarray, float] | None:
        """(probabilities, voiced_ratio) for one window, or None.

        None means the window did not carry enough speech to be worth
        a label - silence is not quiet evidence, it is no evidence,
        and fusion.py drops the modality rather than pooling a
        distribution the model invented for room tone.
        """

        conditioned = self.condition(samples)

        if conditioned is None:
            return None

        clip, voiced_seconds, rms = conditioned

        if voiced_seconds < self.min_voiced_seconds or rms < self.min_rms:
            return None

        torch = self._torch

        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        with self._lock, torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits.float(), dim=-1)[0]

        voiced_ratio = min(
            1.0, voiced_seconds / (self.clip_samples / self.sample_rate)
        )

        return probabilities.cpu().numpy().astype(np.float64), float(voiced_ratio)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_key,
            "display_name": self.display_name,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "window_seconds": self.clip_samples / self.sample_rate,
        }
