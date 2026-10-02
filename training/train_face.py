"""
Fine-tune the facial emotion ViT on the merged corpus, on a local GPU.

    python prepare_face_data.py
    python train_face.py                       # ~30-50 min on an RTX 4060
    python train_face.py --epochs 1 --limit 3000    # quick trial

Produces `vit-emotion-v3.onnx` and an `emotion_config.json` with the
same signature as the deployed v2 model, so installing it is a path
change and nothing else.

What this changes relative to v2
--------------------------------
Same architecture, same seven-class head, same ONNX contract. The
differences are all in the data:

  * three datasets instead of one (FER2013 + AffectNet + RAF-DB), so
    the model sees faces photographed under conditions FER2013 does
    not contain at all;
  * augmentation aimed at the webcam domain - JPEG artefacts, motion
    blur, uneven exposure, occlusion, grayscale;
  * class-balanced loss, because `disgust` is under 2% of the corpus
    and an unweighted loss simply ignores it.

FER2013's test split is held out and never trained on, so the headline
number is directly comparable to the deployed model's 74.4% on the same
images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# See export_face.py: torch's ONNX exporter logs emoji that the Windows
# cp1252 console cannot encode, killing the export after training has
# already finished.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR.parent / "vit-emotion-v2-deployment"

sys.path.insert(0, str(DEPLOY_DIR))

from labels import NUM_SHARED_LABELS, SHARED_LABELS  # noqa: E402

IMAGE_SIZE = 224
MEAN = np.array([0.5, 0.5, 0.5], np.float32)   # matches emotion_config.json
STD = np.array([0.5, 0.5, 0.5], np.float32)

MODEL_ID = "google/vit-base-patch16-224-in21k"

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# AUGMENTATION
# ============================================================
#
# Every item is a distortion a real camera introduces and a curated
# dataset does not. Grayscale is included deliberately: it stops the
# model keying on skin tone, which is both a fairness concern and a
# generalisation one.
# ============================================================


def random_framing(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-frame the face the way a detector crop would.

    Every source here stores a whole dataset image: head, hair, chin and
    some background. Deployment does not hand the model that. It hands
    it a YuNet box plus a margin, which is a much tighter and far more
    variable framing, and the model only ever saw one framing in
    training.

    That gap is what broke 'neutral'. Measured on faces trained on by
    neither model, v3 emitted 'neutral' on 2 of 693 webcam crops (F1
    0.000) while scoring 0.56 neutral F1 on the very same images passed
    in uncropped. The expression had not changed; the framing had. The
    deployment crop was retuned to compensate, which recovered part of
    it, but the real fix is to stop the model depending on a framing it
    cannot count on.

    So: sample a sub-rectangle covering 55-100% of the frame, off
    centre, and resize back. The tight end is tighter than any margin
    the server uses and the loose end is the raw image, so the deployed
    crop lands inside the range rather than outside it.
    """

    height, width = image.shape[:2]

    keep = rng.uniform(0.55, 1.0)

    crop_w = max(8, int(width * keep))
    crop_h = max(8, int(height * keep))

    # Off-centre, because a detector box is not perfectly centred on the
    # face either - it sits where the detector put it.
    x0 = int(rng.integers(0, max(1, width - crop_w + 1)))
    y0 = int(rng.integers(0, max(1, height - crop_h + 1)))

    patch = image[y0:y0 + crop_h, x0:x0 + crop_w]

    if patch.size == 0:
        return image

    return cv2.resize(patch, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:

    # First, because everything below assumes a 224 frame and this is
    # the one that decides what is in it.
    if rng.random() < 0.85:
        image = random_framing(image, rng)

    if rng.random() < 0.5:
        image = cv2.flip(image, 1)

    if rng.random() < 0.7:
        angle = rng.uniform(-15, 15)
        scale = rng.uniform(0.88, 1.12)
        shift = rng.uniform(-0.05, 0.05, 2) * IMAGE_SIZE
        matrix = cv2.getRotationMatrix2D(
            (IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, scale
        )
        matrix[:, 2] += shift
        image = cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE),
                               borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.7:
        image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.75, 1.30),
                                    beta=rng.uniform(-30, 30))

    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), 0)

    if rng.random() < 0.35:
        quality = int(rng.integers(35, 85))
        ok, buffer = cv2.imencode(".jpg", image,
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if rng.random() < 0.20:
        noise = rng.normal(0, 6, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.15:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if rng.random() < 0.10:
        size = int(rng.integers(30, 70))
        x, y = rng.integers(0, IMAGE_SIZE - size, 2)
        image[y:y + size, x:x + size] = int(rng.integers(0, 255))

    return image


def to_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


# ============================================================
# DATA
# ============================================================


class FaceFolder(Dataset):
    """Images in <root>/<label>/*.jpg, decoded on demand."""

    def __init__(self, paths, labels, train: bool):
        self.paths = paths
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))

        if image is None:
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        elif image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        if self.train:
            # Seeded per index so a run is reproducible, but offset by a
            # per-epoch salt so the same image is not augmented
            # identically every epoch.
            rng = np.random.default_rng(
                (index * 2654435761 + random.getrandbits(20)) % (2 ** 32)
            )
            image = augment(image, rng)

        return to_tensor(image), int(self.labels[index])


def index_folder(root: Path):
    paths, labels = [], []

    for label_index, label in enumerate(SHARED_LABELS):
        for path in sorted((root / label).glob("*.jpg")):
            paths.append(path)
            labels.append(label_index)

    return np.array(paths, dtype=object), np.array(labels)


# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()

    logits_all, labels_all = [], []

    for images, targets in loader:
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            out = model(pixel_values=images.to(device, non_blocking=True)).logits
        logits_all.append(out.float().cpu())
        labels_all.append(targets)

    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    predictions = logits.argmax(1)

    f1 = []
    for c in range(NUM_SHARED_LABELS):
        tp = ((predictions == c) & (labels == c)).sum()
        fp = ((predictions == c) & (labels != c)).sum()
        fn = ((predictions != c) & (labels == c)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)

    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": {SHARED_LABELS[c]: float(f1[c])
                         for c in range(NUM_SHARED_LABELS)},
        "logits": logits,
        "labels": labels,
    }


def softmax_np(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def expected_calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    error = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def fit_temperature(logits, labels):
    def nll(t):
        p = softmax_np(logits / t)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    g = (math.sqrt(5) - 1) / 2
    a, b = 0.05, 10.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
        if abs(b - a) < 1e-4:
            break
    return float((a + b) / 2)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the training set, for a trial run.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Directory of a saved checkpoint to continue training from, "
             "e.g. best-vit-emotion-v5. Omit to start from the pretrained "
             "ViT.",
    )
    parser.add_argument(
        "--balance", type=float, default=0.5,
        help="Class-balanced sampling exponent: 0 disables it, 0.5 is a "
             "compromise, 1.0 samples every class equally often.",
    )
    parser.add_argument(
        "--weight-cap", type=float, default=12.0,
        help="Upper bound on the per-class loss weight. The default is "
             "high enough not to bind on this corpus.",
    )
    parser.add_argument("--out", type=Path, default=BASE_DIR / "vit-emotion-v3.onnx")
    parser.add_argument("--cpu", action="store_true")

    arguments = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu"
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}  {properties.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: training on CPU will take hours.")

    # ---------------- data ----------------

    train_root = arguments.data / "train"
    test_root = arguments.data / "fer2013_test"

    if not train_root.is_dir():
        raise SystemExit(f"{train_root} not found. Run prepare_face_data.py first.")

    paths, labels = index_folder(train_root)

    if arguments.limit and arguments.limit < len(paths):
        pick = np.random.default_rng(SEED).permutation(len(paths))[:arguments.limit]
        paths, labels = paths[pick], labels[pick]

    # Stratified 92/8 split. Validation exists to pick the epoch and fit
    # the temperature; FER2013 test is never touched for either.
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(NUM_SHARED_LABELS):
        members = np.where(labels == c)[0]
        rng.shuffle(members)
        cut = max(1, int(0.08 * len(members)))
        val_idx.extend(members[:cut])
        train_idx.extend(members[cut:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    train_ds = FaceFolder(paths[train_idx], labels[train_idx], True)
    val_ds = FaceFolder(paths[val_idx], labels[val_idx], False)

    test_paths, test_labels = index_folder(test_root)
    test_ds = FaceFolder(test_paths, test_labels, False)

    print(f"train {len(train_ds)}  val {len(val_ds)}  "
          f"held-out FER2013 test {len(test_ds)}")

    loader_kwargs = dict(num_workers=arguments.workers,
                         pin_memory=device.type == "cuda",
                         persistent_workers=arguments.workers > 0)

    # Class-balanced sampling.
    #
    # The corpus is 8:1 against disgust, and a weighted loss alone did
    # not fix it: v4 reached 54.5% precision on disgust but only 12.0%
    # recall, predicting it 11 times for 50 true faces. Precision above
    # angry, fear, neutral and surprise with recall far below all of
    # them is the signature of a model that has learnt the class and is
    # simply reluctant to say it - which is a sampling problem, not a
    # separability one.
    #
    # Sampling with probability proportional to 1/count**alpha. alpha=1
    # is full balance and oversamples the 3,346 disgust images about 8x
    # per epoch, which risks memorising them; alpha=0.5 is the usual
    # compromise and still roughly triples their rate.
    if arguments.balance > 0:
        counts_all = np.bincount(labels[train_idx], minlength=NUM_SHARED_LABELS)
        counts_all = np.maximum(counts_all, 1)

        per_class = (1.0 / counts_all) ** arguments.balance
        sample_weights = per_class[labels[train_idx]]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
        )

        expected = per_class * counts_all
        expected = expected / expected.sum()
        print("sampling rate per class (alpha="
              f"{arguments.balance}): "
              + ", ".join(f"{n}={p * 100:.1f}%"
                          for n, p in zip(SHARED_LABELS, expected)))

        train_loader = DataLoader(train_ds, arguments.batch_size,
                                  sampler=sampler, drop_last=True,
                                  **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, arguments.batch_size, shuffle=True,
                                  drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, arguments.batch_size * 2, shuffle=False,
                            **loader_kwargs)
    test_loader = DataLoader(test_ds, arguments.batch_size * 2, shuffle=False,
                             **loader_kwargs)

    # ---------------- model ----------------

    from transformers import ViTForImageClassification

    # --init-from continues an interrupted run from its best epoch
    # instead of restarting from the ImageNet-21k weights. A long run
    # that dies at epoch 3 of 6 otherwise costs an hour to get back to
    # where it already was.
    source = (str(arguments.init_from) if arguments.init_from else MODEL_ID)

    if arguments.init_from:
        print(f"resuming from {source}")

    model = ViTForImageClassification.from_pretrained(
        source,
        num_labels=NUM_SHARED_LABELS,
        id2label={i: n for i, n in enumerate(SHARED_LABELS)},
        label2id={n: i for i, n in enumerate(SHARED_LABELS)},
    ).to(device)

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    counts = np.bincount(labels[train_idx], minlength=NUM_SHARED_LABELS)
    counts = np.maximum(counts, 1)

    raw_weights = counts.sum() / (NUM_SHARED_LABELS * counts)

    # Capping the loss weight was a mistake at 4.0. The reasoning was
    # that a rare class at a high weight would buy recall by firing on
    # everything; measured, the opposite happened - v4 ended at 54.5%
    # precision and 12.0% recall on disgust, i.e. too shy, not too
    # eager. The cap is now a flag and defaults high enough not to bind,
    # with the balance sampler doing most of the work.
    WEIGHT_CAP = arguments.weight_cap

    if raw_weights.max() > WEIGHT_CAP:
        print(f"capping class weights at {WEIGHT_CAP} "
              f"(uncapped max was {raw_weights.max():.1f})")

    class_weights = torch.tensor(
        np.minimum(raw_weights, WEIGHT_CAP), dtype=torch.float32
    ).to(device)

    print("class weights:", {n: round(float(w), 2)
                             for n, w in zip(SHARED_LABELS, class_weights)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr,
                                  weight_decay=0.01)
    total_steps = len(train_loader) * arguments.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=arguments.lr, total_steps=total_steps, pct_start=0.1
    )

    # Label smoothing stands in for the FER+ soft labels: these datasets
    # ship one hard label per image, but the underlying judgements are
    # genuinely uncertain, and training as if they were not makes the
    # model overconfident in exactly the way calibration then has to undo.
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ---------------- train ----------------

    best_f1 = -1.0
    history = []
    # Derived from --out, so two runs cannot overwrite each other's best
    # weights. It used to be the literal string "best-vit-emotion-v3"
    # for every run regardless of what was being trained.
    checkpoint = BASE_DIR / f"best-{arguments.out.stem}"
    started = time.time()

    for epoch in range(1, arguments.epochs + 1):

        model.train()
        running = 0.0
        epoch_start = time.time()

        for step, (images, targets) in enumerate(train_loader, 1):

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                logits = model(pixel_values=images.to(device, non_blocking=True)).logits
                loss = criterion(logits, targets.to(device, non_blocking=True))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()

            if step % 100 == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_start
                remaining = (len(train_loader) - step) * elapsed / step
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f} "
                      f"{elapsed / step:.2f}s/step  eta {remaining / 60:.1f}m",
                      flush=True)

        metrics = evaluate(model, val_loader, device, amp_dtype)

        print(f"  epoch {epoch} validation  acc {metrics['accuracy'] * 100:.2f}%  "
              f"macro-F1 {metrics['macro_f1'] * 100:.2f}%", flush=True)

        history.append({"epoch": epoch, "loss": running / len(train_loader),
                        "val_accuracy": metrics["accuracy"],
                        "val_macro_f1": metrics["macro_f1"]})

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            model.save_pretrained(checkpoint)
            print(f"  saved (best macro-F1 {best_f1 * 100:.2f}%)", flush=True)

    training_seconds = time.time() - started

    # ---------------- test + calibrate ----------------

    model = ViTForImageClassification.from_pretrained(checkpoint).to(device)

    validation = evaluate(model, val_loader, device, amp_dtype)
    temperature = fit_temperature(validation["logits"], validation["labels"])

    test = evaluate(model, test_loader, device, amp_dtype)

    ece_raw = expected_calibration_error(softmax_np(test["logits"]), test["labels"])
    ece_cal = expected_calibration_error(
        softmax_np(test["logits"] / temperature), test["labels"]
    )

    print("\n" + "=" * 68)
    print("HELD-OUT FER2013 TEST  (never seen in training)")
    print("=" * 68)
    print(f"  accuracy   {test['accuracy'] * 100:.2f}%")
    print(f"  macro-F1   {test['macro_f1'] * 100:.2f}%")
    print(f"  temperature {temperature:.3f}   ECE {ece_raw:.4f} -> {ece_cal:.4f}")
    print("\n  per-class F1 (%)")
    for name, value in test["per_class_f1"].items():
        print(f"    {name:<9} {value * 100:6.2f}")
    print("\n  deployed v2 on these same images: 74.40% acc / 73.72% macro-F1")
    print("=" * 68)

    # ---------------- export ----------------

    model.eval().cpu()

    class Wrapper(nn.Module):
        """Unwrap the HF output so ONNX sees a plain logits tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        Wrapper(model), dummy, str(arguments.out),
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18, do_constant_folding=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(arguments.out),
                                   providers=["CPUExecutionProvider"])

    with torch.no_grad():
        torch_out = Wrapper(model)(dummy).numpy()
    onnx_out = session.run(None, {"pixel_values": dummy.numpy()})[0]

    drift = float(np.abs(torch_out - onnx_out).max())
    print(f"\nONNX export: max |torch - onnx| = {drift:.2e}")
    assert drift < 1e-3, "ONNX export diverged from the torch model"
    assert onnx_out.shape == (2, NUM_SHARED_LABELS)

    config = {
        "image_size": IMAGE_SIZE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "temperature": round(temperature, 4),
        "id2label": {str(i): n for i, n in enumerate(SHARED_LABELS)},
        "_provenance": {
            "base_model": MODEL_ID,
            "datasets": "FER2013 train + AffectNet + RAF-DB",
            "train_images": int(len(train_idx)),
            "epochs": arguments.epochs,
            "training_seconds": round(training_seconds, 1),
            "heldout_test": "FER2013 test",
            "test_accuracy": round(test["accuracy"], 4),
            "test_macro_f1": round(test["macro_f1"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in test["per_class_f1"].items()},
            "history": history,
        },
    }

    config_path = arguments.out.with_name("emotion_config.json")
    config_path.write_text(json.dumps(config, indent=4))

    print(f"Wrote {arguments.out}")
    print(f"Wrote {config_path}")
    print("\nInstall into the deployment:")
    print(f"  copy {arguments.out.name} -> ../vit-emotion-v2-deployment/")
    print(f"  copy emotion_config.json -> ../vit-emotion-v2-deployment/")
    print("  update ONNX_PATH in server.py, then run test_deployment.py")


if __name__ == "__main__":
    main()
