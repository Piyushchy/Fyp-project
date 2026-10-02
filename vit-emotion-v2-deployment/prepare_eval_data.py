"""
Build a face evaluation set for eval_face.py.

    python prepare_eval_data.py --per-class 150

Downloads RAF-DB (7 classes, already in the shared label order) and
writes two variants into the class-per-directory layout eval_face.py
expects:

    <out>/clean/<label>/*.jpg    face centred, upright, fixed scale
    <out>/webcam/<label>/*.jpg   face rolled, rescaled, off-centre,
                                 compressed and unevenly lit

Why two variants
----------------
RAF-DB ships pre-cropped, pre-aligned 100x100 faces. Measured on those
directly, alignment has nothing left to correct and would look useless -
which would be a misleading result, not a real one. A webcam does not
hand the model a canonical crop; it hands it a head at an angle, off
centre, at whatever distance the person happens to sit, through a lossy
encoder.

So the perturbation is made the independent variable. `clean` is the
control: alignment should change little, and if it hurts there, the
crop geometry is wrong. `webcam` applies exactly the distortions
alignment exists to undo. Comparing the two isolates the effect
instead of asserting it.

This is a controlled simulation over real labelled faces, and it should
be reported as such - not as a natural in-the-wild webcam benchmark.
The perturbation parameters below are the experiment's definition, so
they are named constants rather than inline literals.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from labels import SHARED_LABELS

# Which labelled face set the frames are built from.
#
# RAF-DB was the original choice and it was the wrong one: training/
# prepare_face_data.py feeds the *same* `train` split into v3, so an
# eval set drawn from it is scored on images the model was fitted on.
# Measured overlap between the RAF-DB eval set and training/data/train
# was 117/120 neutral, 120/120 angry, 120/120 happy. Every v3 number
# produced that way is a memorisation score, not a generalisation one.
#
# `unseen` is the default because it is the only source here that
# neither v2 nor v3 has trained on, which is the condition a stranger
# at a webcam is actually in. RAF-DB is kept selectable so the old
# numbers can be reproduced, but only deliberately.
SOURCES = {
    "unseen": {
        "id": "JasonChen0317/FacialExpressions",
        "split": "train",
        # Already in the shared order, so the mapping is the identity.
        "names": (
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
        ),
        "to_shared": {label: label for label in SHARED_LABELS},
        "trained_on_by": (),
    },
    "rafdb": {
        "id": "deanngkl/raf-db-7emotions",
        "split": "train",
        "names": (
            "anger", "disgust", "fear", "happiness",
            "neutral", "sadness", "surprise",
        ),
        "to_shared": {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "happiness": "happy",
            "neutral": "neutral",
            "sadness": "sad",
            "surprise": "surprise",
        },
        "trained_on_by": ("v3",),
    },
}

DEFAULT_SOURCE = "unseen"

FRAME_W, FRAME_H = 640, 480

# --- control variant ---
CLEAN_FACE_HEIGHT = 250

# --- webcam variant: the distortions alignment is meant to undo ---
WEBCAM_ROLL_DEGREES = 20.0      # head tilt, either direction
WEBCAM_SCALE_RANGE = (0.62, 1.25)   # sitting closer to / further from the camera
WEBCAM_OFFSET_FRACTION = 0.13   # not centred in the frame
WEBCAM_BRIGHTNESS_RANGE = (-38, 38)
WEBCAM_BLUR_PROBABILITY = 0.35
WEBCAM_JPEG_QUALITY = (45, 80)


def composite(
    face_bgr: np.ndarray,
    face_height: int,
    roll_degrees: float,
    centre_x: float,
    centre_y: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paste one face into a full frame at the requested pose."""

    face_height = int(np.clip(face_height, 60, min(FRAME_W, FRAME_H) - 20))

    face = cv2.resize(
        face_bgr, (face_height, face_height), interpolation=cv2.INTER_CUBIC
    )

    # A mid-grey field with a little sensor noise. A perfectly flat
    # background is not something a camera ever produces, and the
    # detector's confidence responds to that.
    frame = np.full((FRAME_H, FRAME_W, 3), 190, np.uint8)
    noise = np.zeros_like(frame)
    cv2.randn(noise, 0, 6)
    frame = cv2.add(frame, noise)

    cx = int(FRAME_W * centre_x)
    cy = int(FRAME_H * centre_y)

    half = face_height // 2
    x0 = int(np.clip(cx - half, 0, FRAME_W - face_height))
    y0 = int(np.clip(cy - half, 0, FRAME_H - face_height))

    # Feathered elliptical paste. A hard square edge gives the detector
    # a rectangle to lock onto and would make detection easier than it
    # has any right to be.
    mask = np.zeros((face_height, face_height), np.float32)
    cv2.ellipse(
        mask,
        (half, half),
        (int(face_height * 0.46), int(face_height * 0.49)),
        0, 0, 360, 1, -1,
    )
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]

    region = frame[y0:y0 + face_height, x0:x0 + face_height].astype(np.float32)
    blended = face.astype(np.float32) * mask + region * (1.0 - mask)
    frame[y0:y0 + face_height, x0:x0 + face_height] = blended.astype(np.uint8)

    if abs(roll_degrees) > 1e-3:
        matrix = cv2.getRotationMatrix2D(
            (x0 + half, y0 + half), roll_degrees, 1.0
        )
        frame = cv2.warpAffine(
            frame, matrix, (FRAME_W, FRAME_H), borderMode=cv2.BORDER_REPLICATE
        )

    return frame


def degrade(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the lossy part of a webcam: exposure, motion blur, JPEG."""

    brightness = float(rng.uniform(*WEBCAM_BRIGHTNESS_RANGE))
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)

    if rng.random() < WEBCAM_BLUR_PROBABILITY:
        kernel = int(rng.choice([3, 5, 7]))
        frame = cv2.GaussianBlur(frame, (kernel, kernel), 0)

    quality = int(rng.integers(*WEBCAM_JPEG_QUALITY))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else frame


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_data",
        help="Destination directory.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=150,
        help="Images per emotion per variant.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default=DEFAULT_SOURCE,
        help=(
            "Which labelled face set to build the frames from. "
            "'unseen' is trained on by neither model; 'rafdb' is v3's "
            "own training data and only reproduces the old numbers."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the destination directory first.",
    )

    arguments = parser.parse_args()

    source = SOURCES[arguments.source]

    if source["trained_on_by"]:
        print(
            f"WARNING: {source['id']} is training data for "
            f"{', '.join(source['trained_on_by'])}. Scores produced "
            f"against it measure memorisation, not generalisation."
        )

    from datasets import load_dataset

    print(f"Loading {source['id']} ({arguments.source}) ...")

    dataset = load_dataset(source["id"], split=source["split"])

    names = tuple(dataset.features["label"].names)

    assert names == source["names"], (
        f"Upstream label order for {source['id']} changed.\n"
        f"  expected {source['names']}\n  observed {names}"
    )

    if arguments.clean and arguments.out.exists():
        shutil.rmtree(arguments.out)

    for variant in ("clean", "webcam"):
        for label in SHARED_LABELS:
            (arguments.out / variant / label).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(arguments.seed)

    # Shuffle once with a fixed seed, then take the first N of each
    # class. Taking them in file order would sample whatever happens to
    # sit at the front of the archive.
    order = rng.permutation(len(dataset))

    written = {label: 0 for label in SHARED_LABELS}

    for position in order:

        if all(count >= arguments.per_class for count in written.values()):
            break

        row = dataset[int(position)]

        label = source["to_shared"][names[row["label"]]]

        if written[label] >= arguments.per_class:
            continue

        face = cv2.cvtColor(
            np.array(row["image"].convert("RGB")), cv2.COLOR_RGB2BGR
        )

        index = written[label]
        stem = f"{label}_{index:05d}.jpg"

        # --- control ---
        clean_frame = composite(
            face, CLEAN_FACE_HEIGHT, 0.0, 0.5, 0.5, rng
        )
        cv2.imwrite(str(arguments.out / "clean" / label / stem), clean_frame)

        # --- webcam ---
        scale = float(rng.uniform(*WEBCAM_SCALE_RANGE))
        roll = float(rng.uniform(-WEBCAM_ROLL_DEGREES, WEBCAM_ROLL_DEGREES))
        offset_x = 0.5 + float(
            rng.uniform(-WEBCAM_OFFSET_FRACTION, WEBCAM_OFFSET_FRACTION)
        )
        offset_y = 0.5 + float(
            rng.uniform(-WEBCAM_OFFSET_FRACTION, WEBCAM_OFFSET_FRACTION)
        )

        webcam_frame = composite(
            face,
            int(CLEAN_FACE_HEIGHT * scale),
            roll,
            offset_x,
            offset_y,
            rng,
        )
        webcam_frame = degrade(webcam_frame, rng)

        cv2.imwrite(str(arguments.out / "webcam" / label / stem), webcam_frame)

        written[label] += 1

    print("\nWritten per class:")
    for label in SHARED_LABELS:
        print(f"  {label:<9} {written[label]:>5}")

    total = sum(written.values())

    print(f"\n{total} frames per variant -> {arguments.out}")
    print("\nNext:")
    print(f"  python eval_face.py --data {arguments.out / 'clean'}")
    print(f"  python eval_face.py --data {arguments.out / 'webcam'}")


if __name__ == "__main__":
    main()
