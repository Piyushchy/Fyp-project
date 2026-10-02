"""
Build evaluation splits that are actually disjoint from what was fitted.

    python make_clean_splits.py

Produces two things.

1. data_v4/fer2013_test_clean
   FER2013's test split with every image that also appears in the
   training corpus removed. FER2013 is known to contain duplicated
   images, and the train and test halves here come from two different
   uploads of it, so 568 of the 7,178 test images - 7.9%, and 28% of
   the 'surprise' class alone - were already in training. Any FER2013
   test number computed before this was measuring partial recall.

2. <eval>/calib and <eval>/test
   A fixed, disjoint split of the unseen evaluation set. Calibration
   fits a temperature and seven bias terms, so scoring it on the rows
   it was fitted on reports the fit, not the model. eval_face.py's
   --fit-calibration already holds out internally, but the configuration
   table and every sweep still ran on the whole set, which means the
   crop geometry and the roll threshold were chosen on rows that later
   reported the result. These directories make the separation physical
   so it cannot be forgotten.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def signature(path: Path) -> str | None:
    """Content hash robust to the resize both copies went through.

    Everything here was written at 224 from a 48x48 FER2013 source, so
    downsampling back to 48 recovers the original pixels closely enough
    for an exact hash to match genuine duplicates without matching
    merely similar faces.
    """

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    small = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

    return hashlib.md5(small.tobytes()).hexdigest()


def dedupe_fer2013_test(data_root: Path) -> None:

    train_root = data_root / "train"
    test_root = data_root / "fer2013_test"
    clean_root = data_root / "fer2013_test_clean"

    if not test_root.is_dir():
        print(f"[skip] {test_root} does not exist")
        return

    print(f"Hashing training corpus under {train_root} ...")

    seen = set()
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in class_dir.iterdir():
            if path.suffix.lower() in IMAGE_SUFFIXES:
                value = signature(path)
                if value:
                    seen.add(value)

    print(f"  {len(seen)} distinct training images")

    if clean_root.exists():
        shutil.rmtree(clean_root)

    kept: Counter = Counter()
    dropped: Counter = Counter()

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue

        (clean_root / class_dir.name).mkdir(parents=True, exist_ok=True)

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            value = signature(path)

            if value is not None and value in seen:
                dropped[class_dir.name] += 1
                continue

            shutil.copy2(path, clean_root / class_dir.name / path.name)
            kept[class_dir.name] += 1

    print(f"\nWrote {clean_root}")
    print(f"{'class':<10}{'kept':>8}{'dropped':>9}{'% dropped':>11}")
    for label in sorted(set(kept) | set(dropped)):
        k, d = kept[label], dropped[label]
        print(f"{label:<10}{k:>8}{d:>9}{d / max(k + d, 1) * 100:>10.1f}%")
    total_k, total_d = sum(kept.values()), sum(dropped.values())
    print(f"{'TOTAL':<10}{total_k:>8}{total_d:>9}"
          f"{total_d / max(total_k + total_d, 1) * 100:>10.1f}%")


def split_eval_set(eval_root: Path, seed: int, test_fraction: float) -> None:
    """Partition an eval set into disjoint calib/ and test/ halves."""

    for variant_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):

        if variant_dir.name in ("calib", "test"):
            continue

        rng = np.random.default_rng(seed)

        calib_root = eval_root / "calib" / variant_dir.name
        test_root = eval_root / "test" / variant_dir.name

        for root in (calib_root, test_root):
            if root.exists():
                shutil.rmtree(root)

        counts = {"calib": Counter(), "test": Counter()}

        for class_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):

            paths = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES
            )

            # Split within each class so both halves stay balanced; a
            # global shuffle would let one class drift between them.
            order = rng.permutation(len(paths))
            cut = int(len(paths) * (1.0 - test_fraction))

            for position, index in enumerate(order):
                target = calib_root if position < cut else test_root
                (target / class_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(paths[int(index)],
                             target / class_dir.name / paths[int(index)].name)
                counts["calib" if position < cut else "test"][class_dir.name] += 1

        print(f"\n{eval_root.name}/{variant_dir.name}: "
              f"calib {sum(counts['calib'].values())} / "
              f"test {sum(counts['test'].values())}")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data_v4")
    parser.add_argument(
        "--eval", type=Path,
        default=BASE_DIR.parent / "vit-emotion-v2-deployment" / "eval_data_unseen",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--skip-dedupe", action="store_true")
    parser.add_argument("--skip-split", action="store_true")

    arguments = parser.parse_args()

    if not arguments.skip_dedupe:
        dedupe_fer2013_test(arguments.data)

    if not arguments.skip_split and arguments.eval.is_dir():
        split_eval_set(arguments.eval, arguments.seed, arguments.test_fraction)


if __name__ == "__main__":
    main()
