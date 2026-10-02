"""
Assemble a multi-dataset training corpus for the face modality.

    python prepare_face_data.py --out data

Merges three public datasets that all annotate the same seven emotions
onto the deployment's shared label space, and writes them to disk as
224x224 JPEGs in class folders:

    FER2013     28,709 train / 7,178 test    48x48 grayscale, noisy labels
    AffectNet   27,823                       in-the-wild, better labels
    RAF-DB      20,471                       in-the-wild, crowd-annotated

Why merge rather than pick one
------------------------------
The deployed model was trained on FER2013 alone. Measured, it scores
74.4% on FER2013's own test split and 51.8% on RAF-DB - a 23-point drop
on faces it has never seen the like of. That gap, not the FER2013
number, is what a user sees on their webcam.

Training across three datasets collected under different conditions
attacks exactly that. FER2013 contributes volume and the distribution
the current model already knows, AffectNet and RAF-DB contribute real
photographs with the lighting, pose and resolution variety FER2013
lacks entirely.

FER2013's *test* split is held out completely and never written here,
so it stays a clean comparison against the deployed model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vit-emotion-v2-deployment"))

from labels import SHARED_LABELS  # noqa: E402

OUTPUT_SIZE = 224

# Each source's own class order, and how it maps onto the shared seven.
# Asserted against the downloaded features so a reordering upstream
# cannot silently relabel a third of the corpus.
SOURCES = {
    # FERPlus is FER2013's images re-annotated by 10 crowd workers and
    # resolved by majority vote. FER2013's original single-annotator
    # labels are the noisiest thing in this corpus, so where the two
    # disagree FERPlus is the one to believe. It is listed first
    # because it is the cleanest source measured (see LABEL_AGREEMENT).
    "ferplus": {
        "id": "deanngkl/ferplus-7cls",
        "split": "train",
        "image_key": "image",
        "label_key": "label",
        "names": ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"),
        "map": {"anger": "angry", "disgust": "disgust", "fear": "fear",
                "happiness": "happy", "neutral": "neutral", "sadness": "sad",
                "surprise": "surprise"},
    },
    "fer2013": {
        "id": "clip-benchmark/wds_fer2013",
        "split": "train",
        "image_key": "jpg",
        "label_key": "cls",
        "names": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
        "map": {n: n for n in SHARED_LABELS},
    },
    # This is already full-resolution AffectNet: sampled source images
    # run 375-627 px square, well above the 224 written here. A
    # higher-resolution pull was investigated and there is nothing to
    # gain from one:
    #
    #   Piro17/affectnethq               7.9 GB, same 27,823 images,
    #                                    but license-gated - anonymous
    #                                    downloads 401 even though the
    #                                    repo metadata is public
    #   Piro17/dataset-affecthqnet-...   56,532 rows, but every sampled
    #                                    image is 48x48, i.e. downscaled
    #                                    to FER2013 resolution - strictly
    #                                    worse than this
    #   Mauregato/affectnet_short        96x96 - also worse
    #
    # So resolution was never the constraint on this source. Label
    # quality was, and that is what FERPlus addresses.
    "affectnet": {
        "id": "deanngkl/affectnet_no_contempt",
        "split": "train",
        "image_key": "image",
        "label_key": "label",
        "names": ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"),
        "map": {"anger": "angry", "disgust": "disgust", "fear": "fear",
                "happiness": "happy", "neutral": "neutral", "sadness": "sad",
                "surprise": "surprise"},
    },
    "rafdb": {
        "id": "deanngkl/raf-db-7emotions",
        "split": "train",
        "image_key": "image",
        "label_key": "label",
        "names": ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"),
        "map": {"anger": "angry", "disgust": "disgust", "fear": "fear",
                "happiness": "happy", "neutral": "neutral", "sadness": "sad",
                "surprise": "surprise"},
    },
}

# Per-(source, class) label agreement, measured by holding out a model
# that never saw the source and asking what it votes for. Not a ground
# truth, but the spread is far too large to be domain shift alone:
# 'happy' holds at 71-88% across every source, so a source whose
# 'angry' folder scores 15% is not merely out of domain.
#
#   folder     ferplus  fer2013  affectnet  rafdb
#   angry        78.4%    68.1%      31.0%  15.4%   <- top vote: NEUTRAL
#   disgust      69.6%    92.7%      17.7%  12.0%
#   neutral      44.1%    70.3%      54.3%  52.4%
#   happy        86.7%    87.7%      71.1%  75.6%
#
# RAF-DB's 'angry' folder draws more 'neutral' votes than 'angry' ones.
# Training on it is what taught v3 to answer 'angry' for a calm face -
# the exact failure that prompted this rebuild.
#
# What is dropped is decided on the *shape* of the disagreement, not on
# the agreement number alone. The judge is in-domain on FER2013, so it
# flatters that source everywhere, and it is independently weak at
# disgust - which makes a low disgust score poor evidence on its own.
#
#   rafdb/angry      15.4%, top vote NEUTRAL (38%)  -> dropped
#   rafdb/disgust    12.0%, top votes fear 186 / sad 173, disgust 4th
#   affectnet/disgust 17.7%, votes spread fear 177 / angry 148 /
#                     disgust 124 - no systematic collapse into one
#                     wrong class, and disgust is the judge's own
#                     weakest class cross-domain. KEPT: excluding it
#                     too left only 686 disgust images in the whole
#                     corpus, which is a worse problem than its noise.
EXCLUDE_PAIRS = {
    ("rafdb", "angry"),
    ("rafdb", "disgust"),
}

# FER2013 test, held out as the headline benchmark.
HELDOUT = {
    "id": "Piro17/fer2013test",
    "split": "train",
    "image_key": "image",
    "label_key": "label",
    "names": ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    "map": {n: n for n in SHARED_LABELS},
}


def to_bgr(image) -> np.ndarray:
    """PIL image (any mode) -> 3-channel BGR uint8."""

    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def write_split(name: str, spec: dict, root: Path, limit: int | None) -> Counter:
    """Write one source into <root>/<label>/ as JPEGs."""

    from datasets import load_dataset

    print(f"\n[{name}] loading {spec['id']} ...", flush=True)

    dataset = load_dataset(spec["id"], split=spec["split"])

    label_key = spec["label_key"]

    feature = dataset.features[label_key]
    observed = tuple(getattr(feature, "names", ()) or ())

    if observed:
        assert observed == spec["names"], (
            f"[{name}] upstream label order changed.\n"
            f"  expected {spec['names']}\n  observed {observed}"
        )
    else:
        # webdataset sources carry a bare integer class with no names.
        print(f"[{name}] no label names in features; trusting the documented order")

    if limit is not None and limit < len(dataset):
        keep = np.random.default_rng(0).permutation(len(dataset))[:limit]
        dataset = dataset.select(keep)

    counts: Counter = Counter()

    for index in range(len(dataset)):

        row = dataset[index]

        shared = spec["map"][spec["names"][int(row[label_key])]]

        if (name, shared) in EXCLUDE_PAIRS:
            counts[f"_excluded_{shared}"] += 1
            continue

        bgr = to_bgr(row[spec["image_key"]])

        # INTER_CUBIC because FER2013 is 48x48 and every one of those
        # images is being upsampled almost five-fold; bilinear leaves
        # it noticeably softer.
        resized = cv2.resize(bgr, (OUTPUT_SIZE, OUTPUT_SIZE),
                             interpolation=cv2.INTER_CUBIC)

        destination = root / shared / f"{name}_{index:06d}.jpg"

        cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

        counts[shared] += 1

        if index and index % 5000 == 0:
            print(f"  {index}/{len(dataset)}", flush=True)

    print(f"[{name}] wrote {sum(counts.values())} images", flush=True)

    return counts


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="Cap each source, for a quick trial run.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCES),
        default=None,
        help=(
            "Only rewrite these sources. Filenames are deterministic per "
            "source, so re-running one overwrites exactly its own images "
            "and leaves the rest of the corpus alone - useful after "
            "changing EXCLUDE_PAIRS for a single source."
        ),
    )
    parser.add_argument(
        "--skip-heldout",
        action="store_true",
        help="Do not rewrite the FER2013 test split.",
    )

    arguments = parser.parse_args()

    # With no --sources, write every source not marked default:False.
    # With --sources, write exactly what was asked for, including the
    # non-default ones.
    if arguments.sources is None:
        selected = {
            name: spec for name, spec in SOURCES.items()
            if spec.get("default", True)
        }
    else:
        selected = {
            name: SOURCES[name] for name in arguments.sources
        }

    if arguments.clean and arguments.out.exists():
        shutil.rmtree(arguments.out)

    train_root = arguments.out / "train"
    test_root = arguments.out / "fer2013_test"

    for root in (train_root, test_root):
        for label in SHARED_LABELS:
            (root / label).mkdir(parents=True, exist_ok=True)

    totals: dict[str, Counter] = {}

    for name, spec in selected.items():
        totals[name] = write_split(name, spec, train_root,
                                   arguments.limit_per_source)

    if not arguments.skip_heldout:
        totals["fer2013_test"] = write_split("fer2013test", HELDOUT, test_root,
                                             arguments.limit_per_source)

    print("\n" + "=" * 70)
    print("TRAINING CORPUS")
    print("=" * 70)
    header = f"{'source':<14}" + "".join(f"{l[:7]:>9}" for l in SHARED_LABELS) + f"{'total':>9}"
    print(header)
    print("-" * 70)

    combined: Counter = Counter()

    for name in selected:
        counts = totals[name]
        combined.update(counts)
        # Excluded rows are tallied under _excluded_<label> keys so the
        # drop is visible; they must not inflate the written total.
        written = sum(v for k, v in counts.items() if not k.startswith("_"))
        row = "".join(f"{counts[l]:>9}" for l in SHARED_LABELS)
        print(f"{name:<14}{row}{written:>9}")

    print("-" * 70)
    row = "".join(f"{combined[l]:>9}" for l in SHARED_LABELS)
    print(f"{'TRAIN TOTAL':<14}{row}{sum(combined.values()):>9}")

    if "fer2013_test" in totals:
        held = totals["fer2013_test"]
        row = "".join(f"{held[l]:>9}" for l in SHARED_LABELS)
        print(f"{'held out':<14}{row}{sum(held.values()):>9}")
    print("=" * 70)

    manifest = {
        "train_total": sum(combined.values()),
        "train_per_class": {l: combined[l] for l in SHARED_LABELS},
        "per_source": {n: dict(totals[n]) for n in totals},
        "heldout": "FER2013 test, never seen in training",
        "image_size": OUTPUT_SIZE,
    }

    (arguments.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {arguments.out}")
    print(f"Next: python train_face.py --data {arguments.out}")


if __name__ == "__main__":
    main()
