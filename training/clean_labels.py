"""
Find per-image label noise by cross-validation, then write a cleaned corpus.

    python clean_labels.py --data data_v4 --out data_v6 --folds 3

Why
---
The single largest measured win in this project came from removing one
mislabelled folder: RAF-DB's 'angry' drew more 'neutral' votes than
'angry' ones, and dropping it was worth about four points. That was
found with a crude audit that could only judge an entire (source,
class) folder at once, and it removed exactly two of them.

Label noise does not arrive in folder-sized units. It is per image, and
the rest of the corpus still carries it. This does the same test at
image granularity.

Method (confident learning, in short)
-------------------------------------
A model cannot judge an image it was fitted on - it has memorised it,
and will agree with whatever label it was given. So the corpus is split
into k folds and k models are trained, each on k-1 folds. Every image is
then scored by the one model that never saw it, which makes the
disagreement meaningful.

An image is dropped when the out-of-fold model gives its stated label a
very low probability AND is confident about some other class. Both
conditions matter: the first alone would delete every hard example,
which is exactly the data worth keeping.

Ensembling v3/v4/v5 was measured first and did not help (52.9% for v4
alone against 51.5% for v3+v4), which says these models fail the same
way rather than independently - a shared-data problem, not a capacity
one. That is the evidence this attacks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "vit-emotion-v2-deployment"))

from labels import NUM_SHARED_LABELS, SHARED_LABELS  # noqa: E402

# Reuse the exact dataset and augmentation the real training uses, so
# the fold models see what the final model will see.
_train_src = (BASE_DIR / "train_face.py").read_text(encoding="utf-8")
_module: dict = {"__name__": "_tf", "__file__": str(BASE_DIR / "train_face.py")}
exec(compile(_train_src.split("def main(")[0], "train_face.py", "exec"), _module)

FaceFolder = _module["FaceFolder"]
MODEL_ID = _module["MODEL_ID"]
IMAGE_SIZE = _module["IMAGE_SIZE"]
set_seed = _module["set_seed"]


def index_corpus(root: Path):
    paths, labels = [], []
    for index, name in enumerate(SHARED_LABELS):
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(path)
                labels.append(index)
    return np.array(paths), np.array(labels)


def train_fold(paths, labels, train_idx, epochs, batch_size, lr, device):

    from transformers import ViTForImageClassification

    dataset = FaceFolder(paths[train_idx], labels[train_idx], train=True)
    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=device.type == "cuda")

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_SHARED_LABELS,
        id2label={i: n for i, n in enumerate(SHARED_LABELS)},
        label2id={n: i for i, n in enumerate(SHARED_LABELS)},
    ).to(device)

    counts = np.maximum(np.bincount(labels[train_idx],
                                    minlength=NUM_SHARED_LABELS), 1)
    weights = torch.tensor(counts.sum() / (NUM_SHARED_LABELS * counts),
                           dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(loader) * epochs, pct_start=0.1)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model.train()
    started = time.time()

    for epoch in range(1, epochs + 1):
        for step, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                logits = model(pixel_values=images.to(device)).logits
                loss = criterion(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 200 == 0:
                elapsed = time.time() - started
                total = len(loader) * epochs
                done = (epoch - 1) * len(loader) + step
                print(f"    epoch {epoch} step {step}/{len(loader)} "
                      f"loss {loss.item():.3f} "
                      f"eta {(total - done) * elapsed / done / 60:.1f}m",
                      flush=True)

    return model


@torch.no_grad()
def predict(model, paths, labels, index, batch_size, device):
    """Probabilities for held-out images, with augmentation off."""

    dataset = FaceFolder(paths[index], labels[index], train=False)
    loader = DataLoader(dataset, batch_size * 2, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.eval()

    out = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            logits = model(pixel_values=images.to(device)).logits
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())

    return np.concatenate(out)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data_v4")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data_v6")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--label-prob", type=float, default=0.10,
        help="Drop when the out-of-fold model gives the stated label less "
             "than this probability...",
    )
    parser.add_argument(
        "--other-prob", type=float, default=0.55,
        help="...and simultaneously gives some other class at least this "
             "much. Both conditions are required so that merely hard "
             "images survive and only confidently-contradicted ones go.",
    )
    parser.add_argument(
        "--scores", type=Path, default=BASE_DIR / "label_scores.npz",
        help="Cache of out-of-fold probabilities, so the corpus can be "
             "re-cut at different thresholds without retraining.",
    )
    parser.add_argument("--reuse-scores", action="store_true")

    arguments = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = index_corpus(arguments.data / "train")
    print(f"corpus {len(paths)} images from {arguments.data / 'train'}")
    print("  " + "  ".join(f"{n}={int((labels == i).sum())}"
                           for i, n in enumerate(SHARED_LABELS)))

    if arguments.reuse_scores and arguments.scores.exists():
        cached = np.load(arguments.scores, allow_pickle=True)
        probabilities = cached["probabilities"]
        print(f"\nreusing cached scores from {arguments.scores}")
    else:
        rng = np.random.default_rng(42)
        order = rng.permutation(len(paths))
        fold_of = np.zeros(len(paths), dtype=int)
        for position, index in enumerate(order):
            fold_of[index] = position % arguments.folds

        probabilities = np.zeros((len(paths), NUM_SHARED_LABELS), np.float32)

        for fold in range(arguments.folds):
            train_idx = np.where(fold_of != fold)[0]
            holdout_idx = np.where(fold_of == fold)[0]

            print(f"\n--- fold {fold + 1}/{arguments.folds}: "
                  f"train {len(train_idx)}  score {len(holdout_idx)} ---",
                  flush=True)

            model = train_fold(paths, labels, train_idx, arguments.epochs,
                               arguments.batch_size, arguments.lr, device)

            probabilities[holdout_idx] = predict(
                model, paths, labels, holdout_idx, arguments.batch_size, device)

            del model
            torch.cuda.empty_cache()

        np.savez_compressed(arguments.scores, probabilities=probabilities,
                            labels=labels,
                            paths=np.array([str(p) for p in paths]))
        print(f"\nwrote {arguments.scores}")

    # ---------------- decide what to drop ----------------

    own = probabilities[np.arange(len(labels)), labels]
    best_other = probabilities.copy()
    best_other[np.arange(len(labels)), labels] = -1.0
    other_value = best_other.max(axis=1)
    other_class = best_other.argmax(axis=1)

    drop = (own < arguments.label_prob) & (other_value >= arguments.other_prob)

    print(f"\nflagged {int(drop.sum())} / {len(drop)} "
          f"({drop.mean() * 100:.1f}%) as mislabelled")
    print(f"  rule: P(stated label) < {arguments.label_prob} "
          f"AND P(some other) >= {arguments.other_prob}")

    print(f"\n{'class':10}{'kept':>8}{'dropped':>9}{'%':>7}   most often really")
    for index, name in enumerate(SHARED_LABELS):
        mask = labels == index
        dropped = int((mask & drop).sum())
        kept = int(mask.sum()) - dropped
        if dropped:
            votes = Counter(other_class[mask & drop])
            top = ", ".join(f"{SHARED_LABELS[k]} {v}"
                            for k, v in votes.most_common(3))
        else:
            top = "-"
        print(f"{name:10}{kept:>8}{dropped:>9}"
              f"{dropped / max(kept + dropped, 1) * 100:>6.1f}%   {top}")

    # by source, since that is how the earlier audit worked
    print(f"\n{'source':12}{'kept':>8}{'dropped':>9}{'%':>7}")
    sources = np.array([p.name.split("_")[0] for p in paths])
    for source in sorted(set(sources)):
        mask = sources == source
        dropped = int((mask & drop).sum())
        kept = int(mask.sum()) - dropped
        print(f"{source:12}{kept:>8}{dropped:>9}"
              f"{dropped / max(kept + dropped, 1) * 100:>6.1f}%")

    # ---------------- write the cleaned corpus ----------------

    train_out = arguments.out / "train"

    if train_out.exists():
        shutil.rmtree(train_out)

    for name in SHARED_LABELS:
        (train_out / name).mkdir(parents=True, exist_ok=True)

    for path, label, flagged in zip(paths, labels, drop):
        if not flagged:
            shutil.copy2(path, train_out / SHARED_LABELS[label] / path.name)

    # Carry the evaluation splits over unchanged.
    for extra in ("fer2013_test", "fer2013_test_clean"):
        source_dir = arguments.data / extra
        if source_dir.is_dir() and not (arguments.out / extra).exists():
            shutil.copytree(source_dir, arguments.out / extra)

    manifest = {
        "source_corpus": str(arguments.data),
        "kept": int((~drop).sum()),
        "dropped": int(drop.sum()),
        "rule": {"label_prob": arguments.label_prob,
                 "other_prob": arguments.other_prob,
                 "folds": arguments.folds,
                 "epochs_per_fold": arguments.epochs},
    }
    (arguments.out / "clean_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {train_out}  ({int((~drop).sum())} images)")
    print(f"Next: python train_face.py --data {arguments.out} "
          f"--epochs 4 --workers 0 --out vit-emotion-v6.onnx")


if __name__ == "__main__":
    main()
