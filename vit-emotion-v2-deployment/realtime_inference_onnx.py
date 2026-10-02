"""
Standalone webcam demo - no server, no browser, just an OpenCV window.

    python realtime_inference_onnx.py                  # GPU if available
    python realtime_inference_onnx.py --device gpu     # refuse to run on CPU
    python realtime_inference_onnx.py --device cpu     # force CPU
    python realtime_inference_onnx.py --model v2       # compare checkpoints

Why this file was rewritten
---------------------------
It used to carry its own copy of the inference path, and that copy had
drifted into every bug the deployment had already fixed:

  * it loaded the v2 weights with v3's emotion_config.json - the wrong
    temperature for the wrong model
  * it hardcoded CPUExecutionProvider, so it could never use a GPU even
    on a machine with a working CUDA install (~46x slower)
  * it used the Haar cascade, which returns no landmarks, so crops
    could not be aligned
  * it cropped the detector box plus 5%, which is far tighter than any
    of these models was trained on and is most of why a calm face used
    to come back 'angry'
  * it applied no logit bias and no temperature at all, so the
    probabilities it printed were the raw uncalibrated ones

None of that was visible from reading it, because it looked like a
working script. It now calls the same FacePipeline the server does, so
the two cannot drift again: fix the pipeline and both paths get it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "v2": ("vit-emotion-v2.onnx", "emotion_config.v2.json"),
    "v3": ("vit-emotion-v3.onnx", "emotion_config.json"),
    "v4": ("vit-emotion-v4.onnx", "emotion_config.v4.json"),
    "v5": ("vit-emotion-v5.onnx", "emotion_config.v5.json"),
    "v6": ("vit-emotion-v6.onnx", "emotion_config.v6.json"),
}


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--device", choices=("auto", "gpu", "cpu"),
        default=os.environ.get("EMOTION_DEVICE", "auto").lower(),
        help="auto: GPU when available, else CPU. gpu: fail if no GPU "
             "provider registers. cpu: force CPU.",
    )
    parser.add_argument(
        "--model", choices=sorted(MODELS),
        default=os.environ.get("EMOTION_MODEL", "v4").lower(),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--no-multicrop", action="store_true",
        help="Score one crop instead of averaging over MULTICROP_RATIOS.",
    )
    arguments = parser.parse_args()

    # Set before importing face_pipeline: the CUDA DLL registration it
    # does on import reads this.
    os.environ["EMOTION_DEVICE"] = arguments.device

    import onnxruntime as ort
    from face_pipeline import (
        CropMode,
        EmotionClassifier,
        FacePipeline,
        build_detector,
        describe_providers,
        select_providers,
    )
    from labels import SHARED_LABELS

    onnx_name, config_name = MODELS[arguments.model]

    with open(BASE_DIR / config_name, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    providers = select_providers(arguments.device != "cpu")

    session = ort.InferenceSession(str(BASE_DIR / onnx_name), providers=providers)
    active = describe_providers(session)

    if active == "CPUExecutionProvider" and arguments.device == "gpu":
        raise SystemExit(
            "--device gpu was requested but no GPU provider registered.\n"
            f"  available: {', '.join(ort.get_available_providers())}\n"
            "  If both onnxruntime and onnxruntime-gpu are installed they\n"
            "  overwrite each other. Fix:\n"
            "    pip uninstall -y onnxruntime onnxruntime-gpu\n"
            "    pip install onnxruntime-gpu"
        )

    classifier = EmotionClassifier(
        session=session,
        input_name=session.get_inputs()[0].name,
        image_size=int(config.get("image_size", 224)),
        mean=np.array(config.get("mean", [0.5] * 3), dtype=np.float32),
        std=np.array(config.get("std", [0.5] * 3), dtype=np.float32),
        temperature=float(config.get("temperature", 1.0)),
        logit_bias=config.get("logit_bias"),
        use_tta=False,
    )

    detector = build_detector()

    pipeline = FacePipeline(
        detector, classifier,
        crop_mode=CropMode.ALIGNED,
        multicrop=not arguments.no_multicrop,
    )
    state = pipeline.session_state()

    print(f"model      {onnx_name}  (T={classifier.temperature:.3f})")
    print(f"provider   {active}")
    print(f"detector   {detector.name}")
    print(f"multicrop  {pipeline.multicrop}")

    if detector.name == "haar":
        print("\nWARNING: Haar fallback - no landmarks, so crops cannot be "
              "aligned and accuracy will be materially lower. "
              "Run: python setup_models.py")

    capture = cv2.VideoCapture(arguments.camera)

    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {arguments.camera}.")

    print("\nPress q to quit.")

    previous = time.time()
    smoothed_fps = 0.0

    try:
        while True:

            ok, frame = capture.read()
            if not ok:
                break

            result = pipeline.process(frame, state)

            now = time.time()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            smoothed_fps = instant if smoothed_fps == 0 else (
                0.9 * smoothed_fps + 0.1 * instant
            )

            if result.detected and result.box is not None:
                box = result.box
                cv2.rectangle(frame, (box.x, box.y),
                              (box.x + box.w, box.y + box.h), (0, 255, 0), 2)

                label = result.label or "..."
                text = f"{label} ({result.confidence * 100:.0f}%)"
                cv2.putText(frame, text, (box.x, max(26, box.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # The full distribution, because a single top-1 label
                # hides exactly the near-ties this model produces.
                if result.smoothed_probabilities is not None:
                    order = np.argsort(-result.smoothed_probabilities)
                    for row, index in enumerate(order[:4]):
                        probability = result.smoothed_probabilities[index]
                        cv2.putText(
                            frame,
                            f"{SHARED_LABELS[index]:<9}{probability * 100:4.0f}%",
                            (12, 120 + row * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                        )

                cv2.putText(frame, f"quality {result.quality.overall * 100:.0f}%",
                            (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (180, 180, 180), 1)
            else:
                cv2.putText(frame, "No face detected", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"{smoothed_fps:.1f} FPS  {active.replace('ExecutionProvider','')}",
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)

            cv2.imshow("Real-Time Facial Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
