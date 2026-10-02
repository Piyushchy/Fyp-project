"""
Fetch the detector asset the face pipeline prefers.

    python setup_models.py

Downloads YuNet (~232 KB) next to this file. YuNet is used instead of
the Haar cascade because it returns a detection confidence and five
facial landmarks; the landmarks are what make the aligned crop in
face_pipeline.py possible, and alignment is the single largest
inference-time accuracy lever available without retraining the ViT.

The download is optional. Without it the pipeline falls back to
OpenCV's bundled Haar cascade, which still runs but produces loose,
unaligned, confidence-free boxes - the configuration that the current
67.7% figure was measured under.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import re
import urllib.error
import urllib.request
from pathlib import Path


# ============================================================
# ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

YUNET_PATH = MODELS_DIR / YUNET_FILENAME

# opencv_zoo tracks its model files with Git LFS, so the raw.github
# URL serves a ~130-byte pointer rather than the model. The pointer is
# useful anyway: it carries the real sha256 and byte size, so fetching
# it first gives a checksum to verify the real download against,
# without hard-coding a hash that upstream could rotate.
YUNET_POINTER_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    "main/models/face_detection_yunet/" + YUNET_FILENAME
)

YUNET_MEDIA_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "main/models/face_detection_yunet/" + YUNET_FILENAME
)

# Sanity bounds for the case where no pointer checksum is available.
YUNET_MIN_BYTES = 100_000
YUNET_MAX_BYTES = 2_000_000


# ============================================================
# DOWNLOAD
# ============================================================

def download(url: str, timeout: float = 60.0) -> bytes:
    """Fetch a URL into memory, so a failed download leaves no partial file."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "emotion-fyp-setup/1.0"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_lfs_pointer(payload: bytes) -> tuple[str, int] | None:
    """Read (sha256, size) out of a Git LFS pointer file, if that is what it is."""

    if len(payload) > 1024:
        return None

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if "git-lfs.github.com/spec" not in text:
        return None

    oid = re.search(r"oid sha256:([0-9a-f]{64})", text)
    size = re.search(r"size (\d+)", text)

    if not oid or not size:
        return None

    return oid.group(1), int(size.group(1))


def fetch_yunet(force: bool = False) -> bool:
    """Ensure the YuNet ONNX file exists. Returns True when it is usable."""

    if YUNET_PATH.exists() and not force:
        size = YUNET_PATH.stat().st_size
        print(f"[ok]   {YUNET_FILENAME} already present ({size / 1024:.0f} KB)")
        return True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- step 1: the pointer, for its checksum ---

    expected: tuple[str, int] | None = None

    try:
        pointer_payload = download(YUNET_POINTER_URL)
        expected = parse_lfs_pointer(pointer_payload)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"[warn] could not read the LFS pointer: {error}")
        pointer_payload = b""

    if expected is None and len(pointer_payload) > YUNET_MIN_BYTES:
        # Upstream stopped using LFS and now serves the model directly.
        payload = pointer_payload
    else:
        # --- step 2: the real file, through the LFS media host ---

        if expected is not None:
            print(
                f"[..]   downloading {YUNET_FILENAME} "
                f"({expected[1] / 1024:.0f} KB, sha256:{expected[0][:16]})"
            )
        else:
            print(f"[..]   downloading {YUNET_FILENAME} from opencv_zoo")

        try:
            payload = download(YUNET_MEDIA_URL)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"[warn] download failed: {error}")
            print("[warn] the pipeline will fall back to the Haar cascade.")
            print("[warn] accuracy will be noticeably lower - see README.")
            return False

    # --- step 3: verify before writing ---

    if expected is not None:
        digest = hashlib.sha256(payload).hexdigest()

        if digest != expected[0] or len(payload) != expected[1]:
            print(
                f"[warn] checksum mismatch: got {len(payload)} bytes "
                f"sha256:{digest[:16]}, expected {expected[1]} bytes "
                f"sha256:{expected[0][:16]}. Refusing to write."
            )
            return False

    elif not YUNET_MIN_BYTES <= len(payload) <= YUNET_MAX_BYTES:
        print(
            f"[warn] unexpected payload size {len(payload)} bytes; "
            f"refusing to write. The URL may now serve an HTML error page."
        )
        return False

    YUNET_PATH.write_bytes(payload)

    print(f"[ok]   wrote {YUNET_PATH} ({len(payload) / 1024:.0f} KB, verified)")

    return True


# ============================================================
# VERIFY
# ============================================================

def verify() -> bool:
    """Load the asset through OpenCV to prove it is actually usable."""

    try:
        import cv2
    except ImportError:
        print("[warn] opencv is not installed; cannot verify.")
        return False

    if not hasattr(cv2, "FaceDetectorYN"):
        print(f"[warn] opencv {cv2.__version__} has no FaceDetectorYN.")
        print("[warn] upgrade with: pip install 'opencv-python>=4.8'")
        return False

    if not YUNET_PATH.exists():
        return False

    try:
        cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320))
    except cv2.error as error:
        print(f"[warn] OpenCV could not load the model: {error}")
        return False

    print("[ok]   OpenCV loaded the YuNet detector successfully")

    return True


# ============================================================
# ENTRY POINT
# ============================================================

# ============================================================
# MODEL WEIGHTS
# ============================================================
#
# The ViT weights live in a GitHub Release rather than in the repository.
# ONNX keeps its tensors in a sibling .onnx.data file, about 328 MB for
# these checkpoints. GitHub rejects any file over 100 MB on push, and Git
# LFS would work but bills every clone of a public repo against a 1 GB per
# month bandwidth quota, so a handful of fetchers would break the download
# for everyone else. Release assets have no quota and a 2 GB file limit.
#
# The small .onnx graph files ARE committed, so a clone is missing only
# the weight blobs.

RELEASE_TAG = "face-model-v4"

RELEASE_BASE = (
    "https://github.com/Piyushchy/Fyp-project/releases/download/"
    + RELEASE_TAG
)

WEIGHTS = {
    "vit-emotion-v4.onnx.data": (True, "v4 - the deployed model"),
    "vit-emotion-v6.onnx.data": (False, "v6 - label-cleaning experiment"),
}

WEIGHT_MIN_BYTES = 100_000_000


def fetch_weights(force: bool = False, include_optional: bool = False) -> bool:
    """Download the ViT weight blobs that are too large to commit."""

    ok = True

    for filename, (required, note) in WEIGHTS.items():

        if not required and not include_optional:
            continue

        destination = BASE_DIR / filename

        if destination.exists() and not force:
            size = destination.stat().st_size
            if size >= WEIGHT_MIN_BYTES:
                print(f"[skip] {filename} already present ({size / 1e6:.0f} MB)")
                continue
            print(f"[warn] {filename} is only {size} bytes - re-downloading")

        url = f"{RELEASE_BASE}/{filename}"
        print(f"[get ] {filename}  ({note})")

        try:
            payload = download(url, timeout=600.0)
        except Exception as error:                       # noqa: BLE001
            print(f"[fail] {filename}: {type(error).__name__}: {error}")
            ok = ok and not required
            continue

        # A 404 from GitHub returns a short HTML body, which would
        # otherwise be written out as a corrupt model file.
        if len(payload) < WEIGHT_MIN_BYTES:
            print(f"[fail] {filename}: got {len(payload)} bytes, expected over "
                  f"{WEIGHT_MIN_BYTES / 1e6:.0f} MB - release asset missing?")
            ok = ok and not required
            continue

        destination.write_bytes(payload)
        print(f"[ok  ] {filename}  ({len(payload) / 1e6:.0f} MB)")

    return ok


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file is already present.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Fetch only the detector, not the ViT weights.",
    )
    parser.add_argument(
        "--all-weights",
        action="store_true",
        help="Also fetch the optional experiment checkpoint (v6).",
    )

    arguments = parser.parse_args()

    fetched = fetch_yunet(force=arguments.force)

    if fetched:
        verify()
    else:
        print("\nDetector setup incomplete - the server will still start, "
              "using Haar.")

    if not arguments.skip_weights:
        print("\n--- model weights ---")
        if not fetch_weights(force=arguments.force,
                             include_optional=arguments.all_weights):
            print("\nThe server cannot start without vit-emotion-v4.onnx.data.")
            print(f"Download it from {RELEASE_BASE}/vit-emotion-v4.onnx.data")
            print("and place it next to server.py.")
            return 1

    print("\nSetup complete. Run:  python server.py --device gpu")

    return 0


if __name__ == "__main__":
    sys.exit(main())
