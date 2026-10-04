"""
Assemble a speech-emotion corpus for the audio modality.

    python prepare_audio_data.py --out data

Pulls four public corpora, maps every one of them onto the shared
seven-class space, filters out clips that carry a label no audio in
them supports, splits by SPEAKER rather than by clip, caps the train
split's over-full classes, and writes the result as 16 kHz mono WAVs
plus a manifest.

    CREMA-D   7442   91 actors, studio, 6 emotions, no surprise
    RAVDESS   1440   24 actors, studio, 8 emotions
    SAVEE      480    4 actors, studio, 7 emotions
    MELD     13708   Friends dialogue, in-the-wild, 7 emotions

Why this script exists
----------------------
The multimodal branch's audio corpus was five classes wide and
8.6:1 skewed:

    happy 1647  sad 1647  angry 1647  fear 1647  surprise 192

with no neutral and no disgust. Neutral is the class a live session
spends most of its time in, so a model without it cannot be idle -
it emits whichever class its prior favours on every silent frame,
and the fusion then has to argue with a confident wrong answer.

Three things are fixed here, and the order matters:

    coverage    all seven classes, from corpora that actually
                record surprise and disgust
    honesty     speaker-disjoint splits, so the reported number is
                not an actor-recognition score
    balance     a train-split cap, applied after splitting so it
                cannot leak

What is deliberately NOT done: no oversampling, no SMOTE, no
synthetic minority clips. Duplicating the surprise rows would not
add information, only weight, and weight belongs in the loss where
it is one tunable number instead of a baked-in data decision.

Runtime is roughly 25 minutes on a warm cache, most of it decode.
Downloads total about 2.6 GB.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from config import (
    CLASS_CAP_PERCENTILE,
    CLIP_SAMPLES,
    DROP_RAVDESS_CALM,
    INCLUDE_RAVDESS_SONG,
    LABEL2ID,
    LABELS,
    MELD_SPLIT_FILES,
    MIN_RMS,
    MIN_VOICED_SECONDS,
    SAMPLE_RATE,
    SEED,
    SILENCE_TOP_DB,
    SOURCES,
    TEST_FRACTION,
    VAL_FRACTION,
)

MANIFEST_COLUMNS = (
    "uid",
    "path",
    "label",
    "label_id",
    "source",
    "domain",
    "speaker",
    "split",
    "duration",
    "voiced_seconds",
    "rms",
    "transcript",
)


# ============================================================
# DECODE AND CONDITION
# ============================================================

def _to_mono_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Collapse channels and resample to SAMPLE_RATE."""

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    samples = samples.astype(np.float32, copy=False)

    if rate != SAMPLE_RATE:
        # Polyphase resampling rather than librosa.resample: this runs
        # on every one of ~23k clips, RAVDESS arrives at 48 kHz, and
        # scipy's rational-factor path is roughly 20x faster than a
        # high-quality sinc for a 3:1 ratio that is exactly rational.
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), SAMPLE_RATE)
        samples = resample_poly(
            samples, SAMPLE_RATE // divisor, int(rate) // divisor
        ).astype(np.float32)

    return samples


def _trim_and_crop(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Strip silence, then take the most energetic CLIP_SAMPLES window.

    Returns the conditioned clip and how many seconds of it were
    above the silence floor before cropping - the figure the quality
    filter judges, since a 4-second clip holding 0.1 s of speech is
    not a 4-second observation.
    """

    import librosa

    trimmed, _ = librosa.effects.trim(samples, top_db=SILENCE_TOP_DB)

    if trimmed.size == 0:
        return trimmed, 0.0

    voiced_seconds = trimmed.size / SAMPLE_RATE

    if trimmed.size <= CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        padded[: trimmed.size] = trimmed
        return padded, voiced_seconds

    # Longest clips: slide a CLIP_SAMPLES window and keep the one with
    # the most energy. Both studio corpora open on room tone, and a
    # fixed head crop would hand a tenth of every input to it.
    hop = SAMPLE_RATE // 4
    energy = np.square(trimmed)
    cumulative = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))

    starts = range(0, trimmed.size - CLIP_SAMPLES + 1, hop)
    best_start = max(
        starts,
        key=lambda start: cumulative[start + CLIP_SAMPLES] - cumulative[start],
    )

    return trimmed[best_start : best_start + CLIP_SAMPLES], voiced_seconds


def condition(
    payload: bytes,
) -> tuple[np.ndarray, float, float, float] | None:
    """Decode one encoded clip into a fixed-length conditioned window.

    Returns (clip, original_duration, voiced_seconds, rms), or None if
    the bytes could not be decoded at all.
    """

    try:
        samples, rate = sf.read(io.BytesIO(payload), dtype="float32")
    except Exception:
        return None

    if samples.size == 0:
        return None

    duration = (samples.shape[0]) / float(rate)

    samples = _to_mono_16k(samples, rate)
    clip, voiced_seconds = _trim_and_crop(samples)

    if clip.size == 0:
        return None

    rms = float(np.sqrt(np.mean(np.square(clip))))

    # Peak-normalise. The corpora differ by well over 20 dB in
    # recording level - SAVEE is hot, MELD's dialogue is quiet under a
    # laugh track - and absolute level is a corpus fingerprint, not an
    # emotion cue. Leaving it in lets the model infer the source and
    # take its label prior for free, which inflates validation and
    # collapses on a webcam mic.
    peak = float(np.abs(clip).max())
    if peak > 0:
        clip = clip / peak * 0.95

    return clip, duration, voiced_seconds, rms


def passes_quality(voiced_seconds: float, rms: float) -> bool:
    """Is there enough audio here to carry the emotion it is labelled?"""

    return voiced_seconds >= MIN_VOICED_SECONDS and rms >= MIN_RMS


# ============================================================
# SPEAKER IDENTITY
# ============================================================
#
# The split is by speaker, so every row needs one. These are the
# only three places a speaker id is recoverable, and each is a
# filename convention rather than a column.
# ============================================================

def cremad_speaker(filename: str) -> str:
    """1068_TIE_ANG_XX.wav -> '1068'."""

    return filename.split("_")[0]


_RAVDESS_FIELDS = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)")


def ravdess_fields(filename: str) -> tuple[str, str, str] | None:
    """03-01-02-01-02-02-06.wav -> (channel, emotion_code, actor)."""

    match = _RAVDESS_FIELDS.match(Path(filename).name)

    if match is None:
        return None

    return match.group(2), match.group(3), match.group(7)


def savee_speaker(filename: str) -> str:
    """DC_a01.wav -> 'DC'."""

    return Path(filename).name.split("_")[0]


# ============================================================
# PARQUET SOURCES
# ============================================================

def iter_parquet_source(key: str, spec: dict):
    """Yield (payload, shared_label, speaker, transcript) for one corpus.

    The audio column is loaded with decode=False and handed to
    soundfile directly. The `datasets` Audio feature decodes through
    torchcodec on 4.x and later, which is an extra heavyweight
    dependency for a job soundfile already does.
    """

    from datasets import Audio, load_dataset

    for config, splits in spec["configs"].items():
        for split in splits:
            dataset = load_dataset(spec["hub_id"], config, split=split)

            # Guard the class order. The integer labels are not used
            # below - the string `emotion` column is - but a reordering
            # upstream would mean the corpus this script was written
            # against is not the one being read.
            feature = dataset.features.get("label")
            if feature is not None and getattr(feature, "names", None):
                observed = tuple(feature.names)
                if observed != spec["names"]:
                    raise ValueError(
                        f"{spec['hub_id']} class order is {observed}, "
                        f"expected {spec['names']}. Update SOURCES in "
                        f"config.py before trusting this corpus."
                    )

            dataset = dataset.cast_column("audio", Audio(decode=False))

            for row in dataset:
                filename = Path(str(row.get("file", ""))).name
                emotion = str(row["emotion"])

                if key == "ravdess":
                    fields = ravdess_fields(filename)

                    if fields is None:
                        continue

                    channel, _code, actor = fields

                    if channel == "02" and not INCLUDE_RAVDESS_SONG:
                        continue

                    if emotion == "calm" and DROP_RAVDESS_CALM:
                        continue

                    speaker = f"ravdess-actor-{actor}"

                elif key == "cremad":
                    speaker = f"cremad-{cremad_speaker(filename)}"

                elif key == "savee":
                    speaker = f"savee-{savee_speaker(filename)}"

                else:
                    speaker = f"{key}-unknown"

                label = spec["map"].get(emotion)

                if label is None:
                    continue

                payload = row["audio"]["bytes"]

                if payload is None:
                    continue

                yield payload, label, speaker, str(row.get("transcription", ""))


# ============================================================
# MELD
# ============================================================
#
# MELD's hub copy is a loading script, which `datasets` no longer
# executes, so the archives are read directly: one CSV of labels per
# split and one tar.gz of FLAC named dia{D}_utt{U}.flac.
#
# The tars are streamed and decoded entry by entry rather than
# extracted to disk - the three of them are 1.5 GB of FLAC that gets
# transcoded to WAV immediately and never read again.
# ============================================================

MELD_BASE = "https://huggingface.co/datasets/{hub}/resolve/main/{name}"


def _fetch(url: str, timeout: float = 600.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "emotion-fyp-audio-prepare/1.0"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_to(url: str, destination: Path, timeout: float = 600.0) -> None:
    """Stream a large file to disk, resuming a partial download.

    The MELD archives are ~1.1 GB each. Reading one into memory and
    then writing it costs twice that in peak RSS and loses the whole
    transfer if anything interrupts it. This writes through a .part
    file and asks for a byte range when one is already there, so a
    dropped connection costs only what had not arrived yet.
    """

    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": "emotion-fyp-audio-prepare/1.0"}

    if have:
        headers["Range"] = f"bytes={have}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        # 416 means the .part is already the whole file, or is longer
        # than the resource; either way, start over rather than guess.
        if error.code == 416 and have:
            partial.unlink()
            have = 0
            response = urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "emotion-fyp-audio-prepare/1.0"}
                ),
                timeout=timeout,
            )
        else:
            raise

    with response:
        resuming = response.status == 206

        if have and not resuming:
            # Server ignored the range header; the body is the whole
            # file, so the bytes already on disk are not a prefix to
            # append to.
            partial.unlink()
            have = 0

        remaining = response.headers.get("Content-Length")
        total = (int(remaining) + have) if remaining else None

        mode = "ab" if have else "wb"
        done = have
        next_mark = done + 100_000_000

        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1 << 20)

                if not chunk:
                    break

                handle.write(chunk)
                done += len(chunk)

                if done >= next_mark:
                    if total:
                        print(
                            f"      {done / 1e6:.0f} / {total / 1e6:.0f} MB",
                            flush=True,
                        )
                    else:
                        print(f"      {done / 1e6:.0f} MB", flush=True)

                    next_mark = done + 100_000_000

    if total and done != total:
        raise OSError(
            f"{destination.name}: got {done} bytes, expected {total}"
        )

    partial.replace(destination)


def meld_labels(hub_id: str, csv_name: str) -> dict[str, tuple[str, str]]:
    """{'dia0_utt0': (emotion, utterance)} for one MELD split."""

    payload = _fetch(MELD_BASE.format(hub=hub_id, name=csv_name))

    # The CSVs carry CP-1252 smart quotes inside otherwise-ASCII text.
    text = payload.decode("utf-8", errors="replace")

    rows = {}

    for row in csv.DictReader(io.StringIO(text)):
        key = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        rows[key] = (row["Emotion"].strip(), row["Utterance"].strip())

    return rows


def iter_meld_split(hub_id: str, split: str, cache_dir: Path):
    """Yield (payload, shared_label, speaker, transcript) for one split."""

    csv_name, tar_name = MELD_SPLIT_FILES[split]

    labels = meld_labels(hub_id, csv_name)
    label_map = SOURCES["meld"]["map"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / Path(tar_name).name

    if not archive.exists():
        print(f"    downloading {tar_name} ...", flush=True)
        _download_to(MELD_BASE.format(hub=hub_id, name=tar_name), archive)

    print(f"    reading {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")

    with gzip.open(archive, "rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                stem = Path(member.name).stem

                if stem not in labels:
                    continue

                emotion, utterance = labels[stem]
                label = label_map.get(emotion)

                if label is None:
                    continue

                handle = tar.extractfile(member)

                if handle is None:
                    continue

                # Speakers are recurring characters across all three
                # splits, so a speaker id would not make the official
                # dialogue split disjoint and is not pretended to.
                yield handle.read(), label, f"meld-{split}", utterance


# ============================================================
# ASSEMBLY
# ============================================================

def write_clip(destination: Path, clip: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, clip, SAMPLE_RATE, subtype="PCM_16")


def collect(out_dir: Path, sources: list[str]) -> tuple[list[dict], dict]:
    """Decode every requested corpus into out_dir/clips, return rows."""

    rows: list[dict] = []
    stats: dict[str, dict] = {}

    for key in sources:
        spec = SOURCES[key]
        print(f"\n--- {key} ({spec['hub_id']}) ---", flush=True)

        kept = 0
        rejected_quality = 0
        rejected_decode = 0

        if key == "meld":
            streams = [
                (split, iter_meld_split(spec["hub_id"], split, out_dir / "_meld"))
                for split in ("train", "validation", "test")
            ]
        else:
            streams = [(None, iter_parquet_source(key, spec))]

        for forced_split, stream in streams:
            for index, (payload, label, speaker, transcript) in enumerate(stream):
                conditioned = condition(payload)

                if conditioned is None:
                    rejected_decode += 1
                    continue

                clip, duration, voiced_seconds, rms = conditioned

                if not passes_quality(voiced_seconds, rms):
                    rejected_quality += 1
                    continue

                uid = f"{key}-{forced_split or 'all'}-{index:06d}"
                relative = Path("clips") / key / label / f"{uid}.wav"

                write_clip(out_dir / relative, clip)

                rows.append(
                    {
                        "uid": uid,
                        "path": relative.as_posix(),
                        "label": label,
                        "label_id": LABEL2ID[label],
                        "source": key,
                        "domain": spec["domain"],
                        "speaker": speaker,
                        # MELD's official split is authoritative and
                        # survives assign_splits untouched.
                        "split": forced_split or "",
                        "duration": round(duration, 3),
                        "voiced_seconds": round(voiced_seconds, 3),
                        "rms": round(rms, 5),
                        "transcript": transcript,
                    }
                )

                kept += 1

                if kept % 1000 == 0:
                    print(f"    {kept} clips", flush=True)

        stats[key] = {
            "kept": kept,
            "rejected_decode": rejected_decode,
            "rejected_quality": rejected_quality,
        }

        print(
            f"    kept {kept}, dropped {rejected_quality} on quality, "
            f"{rejected_decode} undecodable"
        )

    return rows, stats


# ============================================================
# SPEAKER-DISJOINT SPLIT
# ============================================================

def assign_splits(rows: list[dict]) -> None:
    """Fill the empty `split` fields, allocating whole speakers.

    Speakers are sorted by clip count and dealt largest-first into
    whichever of test/validation/train is furthest below its target.
    Greedy largest-first rather than random: CREMA-D's actors differ
    by a factor of three in clip count, and a random draw of 16% of
    the actors routinely lands 11% or 23% of the clips.
    """

    unassigned = [row for row in rows if not row["split"]]

    if not unassigned:
        return

    by_speaker: dict[str, list[dict]] = defaultdict(list)

    for row in unassigned:
        by_speaker[row["speaker"]].append(row)

    total = len(unassigned)
    targets = {
        "test": TEST_FRACTION * total,
        "validation": VAL_FRACTION * total,
        "train": (1.0 - TEST_FRACTION - VAL_FRACTION) * total,
    }
    filled = {name: 0 for name in targets}

    # Shuffle first so that equal-sized speakers are dealt in a seeded
    # but arbitrary order, then sort by size - Python's sort is stable,
    # so the shuffle survives as the tie-break.
    order = list(by_speaker.items())
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda item: -len(item[1]))

    for speaker, speaker_rows in order:
        # Deficit as a fraction of the target, so the three splits
        # compete on equal terms despite very unequal sizes.
        split = max(
            targets,
            key=lambda name: (targets[name] - filled[name]) / targets[name],
        )

        for row in speaker_rows:
            row["split"] = split

        filled[split] += len(speaker_rows)


# ============================================================
# BALANCE
# ============================================================

def cap_train_classes(rows: list[dict]) -> dict:
    """Subsample over-full classes in the train split only.

    The cap is the CLASS_CAP_PERCENTILE'th percentile of the per-class
    counts, so it tracks the corpus instead of being a magic number.
    Over-full classes are thinned proportionally across their source
    corpora, so capping neutral does not delete all of MELD's neutral
    and keep all of CREMA-D's - that would quietly undo the domain mix
    this corpus was assembled for.
    """

    train = [row for row in rows if row["split"] == "train"]
    counts = Counter(row["label"] for row in train)

    if not counts:
        return {"cap": 0, "before": {}, "after": {}, "dropped": 0}

    cap = int(np.percentile(list(counts.values()), CLASS_CAP_PERCENTILE))

    drop: set[str] = set()

    for label, count in counts.items():
        if count <= cap:
            continue

        by_source: dict[str, list[dict]] = defaultdict(list)

        for row in train:
            if row["label"] == label:
                by_source[row["source"]].append(row)

        surplus = count - cap

        # Proportional shares, then hand the rounding remainder to the
        # largest contributors so the totals come out exact.
        shares = {
            source: surplus * len(members) / count
            for source, members in by_source.items()
        }
        quotas = {source: int(share) for source, share in shares.items()}
        remainder = surplus - sum(quotas.values())

        for source in sorted(
            shares, key=lambda name: -(shares[name] - quotas[name])
        )[:remainder]:
            quotas[source] += 1

        for source, members in by_source.items():
            quota = min(quotas[source], len(members))

            if quota <= 0:
                continue

            # Drop the least-voiced clips first. Within a class and a
            # corpus they are the least informative rows available -
            # short, quiet, closest to the quality floor - so thinning
            # from that end costs less than a uniform random draw.
            members.sort(key=lambda row: (row["voiced_seconds"], row["uid"]))

            for row in members[:quota]:
                drop.add(row["uid"])

    kept = [row for row in rows if row["uid"] not in drop]
    rows[:] = kept

    after = Counter(row["label"] for row in rows if row["split"] == "train")

    return {
        "cap": cap,
        "before": dict(counts),
        "after": dict(after),
        "dropped": len(drop),
    }


# ============================================================
# REPORTING
# ============================================================

def distribution_table(rows: list[dict]) -> str:
    splits = ("train", "validation", "test")
    lines = [
        f"{'class':10s} " + " ".join(f"{name:>11s}" for name in splits) + "   total",
        "-" * 58,
    ]

    for label in LABELS:
        cells = []
        total = 0

        for split in splits:
            count = sum(
                1
                for row in rows
                if row["label"] == label and row["split"] == split
            )
            cells.append(f"{count:>11d}")
            total += count

        lines.append(f"{label:10s} " + " ".join(cells) + f"{total:>8d}")

    lines.append("-" * 58)

    cells = [
        f"{sum(1 for row in rows if row['split'] == split):>11d}"
        for split in splits
    ]
    lines.append(f"{'total':10s} " + " ".join(cells) + f"{len(rows):>8d}")

    return "\n".join(lines)


def domain_table(rows: list[dict]) -> str:
    lines = [f"{'source':10s} {'domain':8s} {'train':>8s} {'val':>8s} {'test':>8s}"]

    for source in sorted({row["source"] for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        counts = Counter(row["split"] for row in subset)
        lines.append(
            f"{source:10s} {subset[0]['domain']:8s} "
            f"{counts['train']:>8d} {counts['validation']:>8d} "
            f"{counts['test']:>8d}"
        )

    return "\n".join(lines)


def assert_speaker_disjoint(rows: list[dict]) -> None:
    """Fail loudly if any acted speaker appears in two splits.

    The whole point of the speaker split is that this cannot happen,
    which is exactly why it is worth asserting: a silent leak here
    would inflate every number this module reports and nothing
    downstream could detect it.
    """

    speakers: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        if row["domain"] != "acted":
            continue

        speakers[row["speaker"]].add(row["split"])

    leaked = {name: sorted(s) for name, s in speakers.items() if len(s) > 1}

    if leaked:
        raise AssertionError(f"speakers span multiple splits: {leaked}")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument(
        "--sources",
        default="cremad,ravdess,savee,meld",
        help="comma-separated subset of the corpora to pull",
    )
    parser.add_argument(
        "--keep-meld-archives",
        action="store_true",
        help="keep the downloaded MELD tars instead of deleting them",
    )
    arguments = parser.parse_args()

    sources = [name.strip() for name in arguments.sources.split(",") if name.strip()]

    unknown = [name for name in sources if name not in SOURCES]

    if unknown:
        print(f"unknown sources: {unknown}; known: {sorted(SOURCES)}")
        return 2

    out_dir = Path(arguments.out)

    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("ASSEMBLING THE AUDIO CORPUS")
    print("=" * 62)
    print(f"  out            {out_dir}")
    print(f"  sources        {', '.join(sources)}")
    print(f"  shared labels  {', '.join(LABELS)}")
    print(f"  clip           {CLIP_SAMPLES / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz mono")

    rows, source_stats = collect(out_dir, sources)

    if not rows:
        print("\nnothing collected")
        return 1

    print("\n--- splitting by speaker ---")
    assign_splits(rows)
    assert_speaker_disjoint(rows)

    acted_speakers = {
        row["speaker"] for row in rows if row["domain"] == "acted"
    }
    print(f"  {len(acted_speakers)} acted speakers, none spanning two splits")

    print("\n--- distribution before capping ---")
    print(distribution_table(rows))

    balance = cap_train_classes(rows)

    print(f"\n--- capping train classes at p{CLASS_CAP_PERCENTILE:.0f} = {balance['cap']} ---")
    print(f"  dropped {balance['dropped']} train clips")
    print(distribution_table(rows))

    print("\n--- by source ---")
    print(domain_table(rows))

    manifest = out_dir / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_rate": SAMPLE_RATE,
        "clip_samples": CLIP_SAMPLES,
        "labels": list(LABELS),
        "sources": source_stats,
        "ravdess_calm_dropped": DROP_RAVDESS_CALM,
        "ravdess_song_included": INCLUDE_RAVDESS_SONG,
        "quality_filter": {
            "min_voiced_seconds": MIN_VOICED_SECONDS,
            "min_rms": MIN_RMS,
            "silence_top_db": SILENCE_TOP_DB,
        },
        "splits": dict(Counter(row["split"] for row in rows)),
        "class_counts": {
            split: {
                label: sum(
                    1
                    for row in rows
                    if row["split"] == split and row["label"] == label
                )
                for label in LABELS
            }
            for split in ("train", "validation", "test")
        },
        "domain_counts": {
            domain: dict(
                Counter(
                    row["split"] for row in rows if row["domain"] == domain
                )
            )
            for domain in sorted({row["domain"] for row in rows})
        },
        "balance": balance,
        "acted_speakers": len(acted_speakers),
        "total_clips": len(rows),
    }

    (out_dir / "corpus.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not arguments.keep_meld_archives:
        archives = out_dir / "_meld"

        if archives.exists():
            shutil.rmtree(archives, ignore_errors=True)
            print("\n  removed the MELD tars (--keep-meld-archives to keep them)")

    print(f"\nwrote {manifest}")
    print(f"wrote {out_dir / 'corpus.json'}")
    print(f"\n{len(rows)} clips ready. Train with:\n  python train_audio.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
