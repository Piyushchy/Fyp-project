"""
Old model vs new model, on the same images, with the same preprocessing.

    python compare_models.py

Two test sets, and the distinction between them matters:

  FER2013 test   Held out from v3's training, and it is v2's home
                 turf. Tells you whether the retrain cost anything on
                 the distribution the old model was built for.

  A fourth dataset  Never seen by EITHER model. This is the only
                 measurement here that speaks to generalisation, and
                 it is the one worth quoting.

Why the second one is necessary
-------------------------------
v3 trains on FER2013 + AffectNet + RAF-DB. That is deliberate - the
whole point was to stop the model knowing only FER2013 - but it means
RAF-DB can no longer serve as v3's cross-dataset check the way it did
for v2. Reusing it would compare a training set against a test set and
report the difference as progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR.parent / "vit-emotion-v2-deployment"

sys.path.insert(0, str(DEPLOY_DIR))

from face_pipeline import select_providers  # noqa: E402
from labels import NUM_SHARED_LABELS, SHARED_LABELS  # noqa: E402

MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)

# A fourth 7-class set, in our exact label order, used by neither model.
UNSEEN = {
    "id": "JasonChen0317/FacialExpressions",
    "split": "train",
    "names": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
}


def preprocess(bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(bgr, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def scores(session, images, labels) -> dict:
    name = session.get_inputs()[0].name

    predictions = []
    for start in range(0, len(images), 64):
        batch = np.stack([preprocess(i) for i in images[start:start + 64]])
        predictions.append(
            session.run(None, {name: batch.astype(np.float32)})[0].argmax(1)
        )

    predicted = np.concatenate(predictions)

    f1 = []
    for c in range(NUM_SHARED_LABELS):
        tp = ((predicted == c) & (labels == c)).sum()
        fp = ((predicted == c) & (labels != c)).sum()
        fn = ((predicted != c) & (labels == c)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)

    return {
        "accuracy": float((predicted == labels).mean()) * 100,
        "macro_f1": float(np.mean(f1)) * 100,
        "per_class_f1": {SHARED_LABELS[c]: float(f1[c]) * 100
                         for c in range(NUM_SHARED_LABELS)},
        "n": len(labels),
    }


def load_folder(root: Path, limit: int | None = None):
    images, labels = [], []
    for index, label in enumerate(SHARED_LABELS):
        paths = sorted((root / label).glob("*.jpg"))
        if limit:
            paths = paths[:limit]
        for path in paths:
            image = cv2.imread(str(path))
            if image is not None:
                images.append(image)
                labels.append(index)
    return images, np.array(labels)


def load_unseen():
    from datasets import load_dataset

    dataset = load_dataset(UNSEEN["id"], split=UNSEEN["split"])

    observed = tuple(dataset.features["label"].names)
    assert observed == UNSEEN["names"], (
        f"label order changed: {observed}"
    )

    images, labels = [], []
    for row in dataset:
        images.append(
            cv2.cvtColor(np.array(row["image"].convert("RGB")), cv2.COLOR_RGB2BGR)
        )
        labels.append(int(row["label"]))

    return images, np.array(labels)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEPLOY_DIR / "vit-emotion-v2.onnx")
    parser.add_argument("--new", type=Path, default=BASE_DIR / "vit-emotion-v3.onnx")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--limit", type=int, default=None)

    arguments = parser.parse_args()

    if not arguments.new.exists():
        raise SystemExit(f"{arguments.new} not found. Run train_face.py first.")

    providers = select_providers(True)

    models = {
        "v2 (FER2013 only)": ort.InferenceSession(str(arguments.old), providers=providers),
        "v3 (3 datasets)": ort.InferenceSession(str(arguments.new), providers=providers),
    }

    suites = {}

    print("Loading FER2013 test (held out from v3, v2's home turf) ...", flush=True)
    suites["FER2013 test"] = load_folder(arguments.data / "fer2013_test", arguments.limit)

    print(f"Loading {UNSEEN['id']} (unseen by BOTH) ...", flush=True)
    try:
        suites["unseen 4th set"] = load_unseen()
    except Exception as error:
        print(f"  could not load: {error}")

    results = {}

    for suite_name, (images, labels) in suites.items():

        print(f"\n{'=' * 66}")
        print(f"{suite_name}   ({len(labels)} images)")
        print("=" * 66)
        print(f"{'model':<22} {'accuracy':>10} {'macro-F1':>10}")
        print("-" * 66)

        results[suite_name] = {}

        for model_name, session in models.items():
            outcome = scores(session, images, labels)
            results[suite_name][model_name] = outcome
            print(f"{model_name:<22} {outcome['accuracy']:>9.2f}% "
                  f"{outcome['macro_f1']:>9.2f}%", flush=True)

        old_r = results[suite_name]["v2 (FER2013 only)"]
        new_r = results[suite_name]["v3 (3 datasets)"]

        print("-" * 66)
        print(f"{'delta':<22} {new_r['accuracy'] - old_r['accuracy']:>+9.2f}  "
              f"{new_r['macro_f1'] - old_r['macro_f1']:>+9.2f}")

        print(f"\n{'per-class F1':<12}" + "".join(f"{l[:7]:>9}" for l in SHARED_LABELS))
        for model_name in models:
            row = "".join(
                f"{results[suite_name][model_name]['per_class_f1'][l]:>9.1f}"
                for l in SHARED_LABELS
            )
            print(f"{model_name[:12]:<12}{row}")

    (BASE_DIR / "results_model_comparison.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nWrote {BASE_DIR / 'results_model_comparison.json'}")


if __name__ == "__main__":
    main()
