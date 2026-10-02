# ============================================================
# SHARED LABEL SPACE
# ============================================================
#
# The single vocabulary both modalities and the fusion module
# agree on. It is the ViT V2 label set, in the exact order the
# ONNX classifier head emits, because that order is baked into
# the exported weights and cannot be changed without re-export.
#
#   emotion_config.json  ->  id2label  ->  this list
#
# The text modality is projected onto this space (see the
# GoEmotions task in ../text-emotion-module/config.py). The
# older six-class MTEB checkpoints reach it through
# TEXT_TO_SHARED_LABEL, which covers only five of the seven -
# they cannot express 'neutral' or 'disgust'.
# ============================================================

from __future__ import annotations


# ============================================================
# VOCABULARY
# ============================================================

SHARED_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

NUM_SHARED_LABELS = len(SHARED_LABELS)

SHARED_ID2LABEL: dict[int, str] = dict(enumerate(SHARED_LABELS))

SHARED_LABEL2ID: dict[str, int] = {
    label: index for index, label in SHARED_ID2LABEL.items()
}


# ============================================================
# PRESENTATION
# ============================================================
#
# Kept here rather than in the frontend so the CLI evaluation
# scripts and the web UI label the same class the same way.
# ============================================================

EMOJI: dict[str, str] = {
    "angry": "\U0001F620",
    "disgust": "\U0001F922",
    "fear": "\U0001F628",
    "happy": "\U0001F60A",
    "neutral": "\U0001F610",
    "sad": "\U0001F622",
    "surprise": "\U0001F632",
}


# ============================================================
# VALIDATION
# ============================================================

def assert_matches_config(id2label: dict[int, str]) -> None:
    """Fail loudly if a model's head disagrees with the shared space.

    Called at startup with the ONNX config. A silent mismatch here
    would misattribute every probability in the fusion pool - class 2
    of one model lining up against class 2 of another that means
    something else - and nothing downstream could detect it.
    """

    observed = tuple(id2label[index] for index in sorted(id2label))

    if observed != SHARED_LABELS:
        raise ValueError(
            f"Model label order {observed} does not match the shared "
            f"space {SHARED_LABELS}. Fusion would compare different "
            f"classes against each other."
        )


def to_vector(mapping: dict[str, float]) -> list[float]:
    """Order a {label: probability} dict into shared-space order."""

    return [float(mapping.get(label, 0.0)) for label in SHARED_LABELS]


def to_mapping(vector) -> dict[str, float]:
    """Name the entries of a shared-space probability vector."""

    return {
        label: float(vector[index])
        for index, label in enumerate(SHARED_LABELS)
    }
