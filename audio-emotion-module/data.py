# ============================================================
# AUDIO EMOTION MODULE - DATA
# ============================================================
#
# Reads the manifest written by prepare_audio_data.py and turns
# it into PyTorch batches.
#
# Two input representations, one per model kind:
#
#   waveform   float32 [CLIP_SAMPLES], for the SSL encoders
#   mfcc       float32 [1, N_MFCC, MFCC_FRAMES], for the baseline
#
# MFCCs are computed once into a cache file rather than per
# epoch. librosa manages about 30 clips a second, which would
# cost more per epoch than the CNN's forward and backward passes
# combined - the baseline would end up measuring librosa.
# ============================================================

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CLIP_SAMPLES,
    DATA_DIR,
    LABELS,
    MANIFEST_PATH,
    MFCC_FRAMES,
    MFCC_HOP_LENGTH,
    N_MFCC,
    NUM_LABELS,
    SAMPLE_RATE,
)


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(path: Path | None = None) -> list[dict]:
    """Read manifest.csv, coercing the numeric columns."""

    path = path or MANIFEST_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the corpus first:\n"
            f"  python prepare_audio_data.py"
        )

    rows = []

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label_id"] = int(row["label_id"])
            row["duration"] = float(row["duration"])
            row["voiced_seconds"] = float(row["voiced_seconds"])
            row["rms"] = float(row["rms"])
            rows.append(row)

    return rows


def split_rows(
    rows: list[dict], split: str, domain: str | None = None
) -> list[dict]:
    """Rows of one split, optionally restricted to one domain."""

    selected = [row for row in rows if row["split"] == split]

    if domain is not None:
        selected = [row for row in selected if row["domain"] == domain]

    return selected


def class_counts(rows: list[dict]) -> np.ndarray:
    """Per-class clip counts in shared-label order."""

    counter = Counter(row["label_id"] for row in rows)

    return np.array(
        [counter.get(index, 0) for index in range(NUM_LABELS)], dtype=np.int64
    )


def class_weights(rows: list[dict], power: float) -> torch.Tensor:
    """(N / count) ** power, normalised to mean 1.

    Normalising matters: an unnormalised inverse-frequency weight
    multiplies the effective learning rate by whatever the mean weight
    happens to be, so the same configured LR would mean different
    things on two different corpora.
    """

    counts = class_counts(rows).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# AUGMENTATION
# ============================================================
#
# Train-time only, and deliberately limited to transforms that
# leave the emotion label true.
#
#   gain        a microphone's distance is not an emotion. Level
#               is also a corpus fingerprint, which the prepare
#               step already peak-normalised away; jittering it
#               keeps the model from relearning the residue.
#   noise       the deployment input is a laptop mic in a room.
#               Every clip here was recorded better than that.
#   time shift  where in the window the utterance starts is an
#               artefact of how the clip was cut.
#
# What is NOT applied: pitch shifting and time stretching. Both
# are standard in speech augmentation and both are wrong here.
# Mean pitch and speaking rate are two of the strongest cues
# separating anger from sadness, so perturbing them moves the
# clip towards another class while keeping its old label - it
# manufactures label noise in the name of robustness.
# ============================================================

class Augment:
    """Waveform augmentation with a per-worker random stream."""

    def __init__(
        self,
        gain_db: float = 6.0,
        noise_snr_db: tuple[float, float] = (10.0, 30.0),
        shift_seconds: float = 0.3,
        probability: float = 0.5,
    ) -> None:
        self.gain_db = gain_db
        self.noise_snr_db = noise_snr_db
        self.shift_samples = int(shift_seconds * SAMPLE_RATE)
        self.probability = probability
        self._rng = np.random.default_rng()

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        rng = self._rng

        if rng.random() < self.probability:
            gain = 10.0 ** (rng.uniform(-self.gain_db, self.gain_db) / 20.0)
            clip = clip * gain

        if rng.random() < self.probability:
            signal_power = float(np.mean(np.square(clip)))

            if signal_power > 0:
                snr = rng.uniform(*self.noise_snr_db)
                noise_power = signal_power / (10.0 ** (snr / 10.0))
                clip = clip + rng.normal(
                    0.0, np.sqrt(noise_power), size=clip.shape
                ).astype(np.float32)

        if rng.random() < self.probability and self.shift_samples > 0:
            shift = int(rng.integers(-self.shift_samples, self.shift_samples + 1))
            clip = np.roll(clip, shift)

            # Zero the wrapped tail rather than leaving it: a circular
            # roll splices the end of the utterance onto its start,
            # which is an acoustic event that does not occur.
            if shift > 0:
                clip[:shift] = 0.0
            elif shift < 0:
                clip[shift:] = 0.0

        peak = float(np.abs(clip).max())

        if peak > 1.0:
            clip = clip / peak

        return clip.astype(np.float32, copy=False)


# ============================================================
# WAVEFORM DATASET
# ============================================================

def read_clip(root: Path, relative: str) -> np.ndarray:
    """Read one conditioned clip, padded or trimmed to CLIP_SAMPLES."""

    samples, _ = sf.read(root / relative, dtype="float32", always_2d=False)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if samples.shape[0] < CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        padded[: samples.shape[0]] = samples
        return padded

    return samples[:CLIP_SAMPLES].astype(np.float32, copy=False)


class WaveformSplit(Dataset):
    """Raw 16 kHz clips for the SSL encoders."""

    def __init__(
        self,
        rows: list[dict],
        root: Path | None = None,
        augment: Augment | None = None,
    ) -> None:
        self.rows = rows
        self.root = root or DATA_DIR
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        clip = read_clip(self.root, row["path"])

        if self.augment is not None:
            clip = self.augment(clip)

        return torch.from_numpy(clip), row["label_id"]


# ============================================================
# MFCC CACHE
# ============================================================

def mfcc_of(clip: np.ndarray) -> np.ndarray:
    """40 MFCCs, per-coefficient normalised. Matches audio_cnn.py."""

    import librosa

    coefficients = librosa.feature.mfcc(
        y=clip,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=MFCC_HOP_LENGTH,
    )

    if coefficients.shape[1] < MFCC_FRAMES:
        coefficients = np.pad(
            coefficients,
            ((0, 0), (0, MFCC_FRAMES - coefficients.shape[1])),
            mode="constant",
        )
    else:
        coefficients = coefficients[:, :MFCC_FRAMES]

    mean = coefficients.mean(axis=1, keepdims=True)
    deviation = coefficients.std(axis=1, keepdims=True) + 1e-8

    return ((coefficients - mean) / deviation).astype(np.float32)


def build_mfcc_cache(rows: list[dict], split: str, root: Path | None = None) -> Path:
    """Compute and store MFCCs for one split. Idempotent."""

    root = root or DATA_DIR
    destination = root / f"mfcc_{split}.npy"
    index_path = root / f"mfcc_{split}.uids.txt"

    uids = [row["uid"] for row in rows]

    if destination.exists() and index_path.exists():
        cached = index_path.read_text(encoding="utf-8").split()

        if cached == uids:
            return destination

    features = np.zeros((len(rows), N_MFCC, MFCC_FRAMES), dtype=np.float32)

    for position, row in enumerate(rows):
        features[position] = mfcc_of(read_clip(root, row["path"]))

        if (position + 1) % 2000 == 0:
            print(f"    mfcc {split} {position + 1}/{len(rows)}", flush=True)

    np.save(destination, features)
    index_path.write_text("\n".join(uids), encoding="utf-8")

    return destination


class MfccSplit(Dataset):
    """Cached MFCC matrices for the baseline CNN."""

    def __init__(
        self,
        rows: list[dict],
        split: str,
        root: Path | None = None,
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.labels = np.array([row["label_id"] for row in rows], dtype=np.int64)
        self.augment = augment
        self._rng = np.random.default_rng()

        path = build_mfcc_cache(rows, split, root)
        self.features = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        features = np.array(self.features[index], dtype=np.float32)

        if self.augment:
            # SpecAugment on the cached matrix. The waveform
            # augmentations cannot be applied here without recomputing
            # the MFCCs, which is the cost the cache exists to avoid;
            # masking the matrix directly buys the same regularisation.
            rng = self._rng

            for _ in range(2):
                width = int(rng.integers(0, max(2, MFCC_FRAMES // 8)))

                if width:
                    start = int(rng.integers(0, MFCC_FRAMES - width))
                    features[:, start : start + width] = 0.0

            width = int(rng.integers(0, max(2, N_MFCC // 8)))

            if width:
                start = int(rng.integers(0, N_MFCC - width))
                features[start : start + width, :] = 0.0

        return torch.from_numpy(features).unsqueeze(0), int(self.labels[index])


# ============================================================
# LOADERS
# ============================================================

def build_loaders(
    rows: list[dict],
    kind: str,
    batch_size: int,
    num_workers: int = 0,
    root: Path | None = None,
):
    """(train, validation, test) loaders for one model kind."""

    splits = {
        name: split_rows(rows, name)
        for name in ("train", "validation", "test")
    }

    loaders = {}

    for name, split in splits.items():
        is_train = name == "train"

        if kind == "ssl":
            dataset: Dataset = WaveformSplit(
                split, root=root, augment=Augment() if is_train else None
            )
        elif kind == "mfcc-cnn":
            dataset = MfccSplit(split, name, root=root, augment=is_train)
        else:
            raise ValueError(f"unknown model kind {kind!r}")

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train and len(dataset) > batch_size,
            persistent_workers=num_workers > 0,
        )

    return loaders["train"], loaders["validation"], loaders["test"], splits


def describe(rows: list[dict]) -> str:
    """One line per split: size, and the class counts behind it."""

    lines = []

    for name in ("train", "validation", "test"):
        split = split_rows(rows, name)
        counts = class_counts(split)
        detail = "  ".join(
            f"{label}:{count}" for label, count in zip(LABELS, counts)
        )
        lines.append(f"  {name:11s} {len(split):6d}   {detail}")

    return "\n".join(lines)
