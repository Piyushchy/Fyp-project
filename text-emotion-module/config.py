# ============================================================
# TEXT EMOTION MODULE - CONFIGURATION
# ============================================================
#
# Central registry for the text modality of the multimodal
# emotion recognition system.
#
# Dataset : MTEB / EmotionClassification  (hub id: mteb/emotion)
# Models  : TinyBERT, DistilBERT, MobileBERT
#
# BERT-base and RoBERTa are deliberately EXCLUDED. Both are
# ~110M / ~125M parameters, which is too heavy for the
# real-time CPU deployment target already established by the
# ViT V2 visual module.
# ============================================================

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODULE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = MODULE_DIR / "checkpoints"
RESULTS_DIR = MODULE_DIR / "results"
ASSETS_DIR = MODULE_DIR / "assets"

RESULTS_FILE = RESULTS_DIR / "benchmark.json"


# ============================================================
# DATASET
# ============================================================

DATASET_ID = "mteb/emotion"

# MTEB EmotionClassification label order.
ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

LABEL2ID = {label: index for index, label in ID2LABEL.items()}

NUM_LABELS = len(ID2LABEL)


# ============================================================
# FUSION LABEL BRIDGE
# ============================================================
#
# The ViT V2 visual module predicts 7 classes:
#   angry, disgust, fear, happy, neutral, sad, surprise
#
# MTEB emotion has 6 text classes. This map projects the text
# labels onto the shared vocabulary used by the fusion module.
# "love" is folded into "happy" because the visual label set
# has no affection class.
# ============================================================

TEXT_TO_SHARED_LABEL = {
    "sadness": "sad",
    "joy": "happy",
    "love": "happy",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprise",
}


# ============================================================
# MODEL REGISTRY
# ============================================================
#
# max_length 128 truncates nothing: the longest training text
# is 87 word-piece tokens and the 99th percentile is 57.
# Batches are padded to their own longest row, not to 128.
#
# TinyBERT gets the highest learning rate and the most epochs.
# It is only generally distilled (not task-distilled on this
# dataset), so its 14M parameters need longer to converge.
#
# MobileBERT gets the HIGHEST learning rate, which is the
# opposite of what its reputation suggests. Its classifier sits
# behind a freshly initialised BatchNorm (see modeling.py for
# why it needs one), and the no_norm backbone makes the
# effective gradient small. Measured on a 3200-row probe,
# 2 epochs:
#
#   lr 3e-5  ->  loss 1.803, macro-F1 0.089   (barely moves)
#   lr 1e-4  ->  loss 1.291, macro-F1 0.118
#   lr 3e-4  ->  loss 1.311, macro-F1 0.126
#
# 1e-4 is chosen over 3e-4: they finish level on the short probe,
# and the lower rate leaves more headroom over the full run's
# 10x longer schedule.
# ============================================================

MODELS = {
    "tinybert": {
        "hub_id": "huawei-noah/TinyBERT_General_4L_312D",
        "display_name": "TinyBERT (4L-312D)",
        "learning_rate": 5e-5,
        "batch_size": 32,
        "epochs": 5,
        "max_length": 128,
        "note": "General distillation of BERT-base: 4 layers, 312 hidden.",
    },
    "distilbert": {
        "hub_id": "distilbert-base-uncased",
        "display_name": "DistilBERT (6L-768D)",
        "learning_rate": 3e-5,
        "batch_size": 32,
        "epochs": 3,
        "max_length": 128,
        "note": "Distilled BERT-base: 6 layers, retains ~97% of GLUE.",
    },
    "mobilebert": {
        "hub_id": "google/mobilebert-uncased",
        "display_name": "MobileBERT (24L-512D)",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 5,
        "max_length": 128,
        "note": "Bottleneck architecture: 24 thin layers, inverted residuals.",
    },
}

# Reference point for the size/speed comparison. Not trained -
# it is the model the three candidates are distilled from.
BERT_BASE_PARAMS = 109_483_778


# ============================================================
# TRAINING DEFAULTS
# ============================================================

SEED = 42
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0


# ============================================================
# HELPERS
# ============================================================

def checkpoint_path(model_key):
    """Directory holding the fine-tuned weights for one model."""

    return CHECKPOINT_DIR / model_key


def ensure_directories():
    """Create the output directories if they do not exist yet."""

    for directory in (CHECKPOINT_DIR, RESULTS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# TASK 2 - GOEMOTIONS, PROJECTED ONTO THE SHARED SEVEN
# ============================================================
#
# Why a second task at all
# ------------------------
# MTEB emotion has six classes and no 'neutral'. Most typed
# messages in a live session are affectively neutral, so a
# six-class model asserts an emotion on every one of them - it
# has no way to say "nothing in particular". It also has no
# 'disgust', which the visual module does have. Fusing a
# six-class text vector against a seven-class face vector means
# two of the seven can only ever be argued for by one modality.
#
# GoEmotions fixes both: it has 'neutral' and 'disgust', and its
# own published Ekman grouping collapses the 28 fine-grained
# labels onto exactly the six basic emotions. Adding 'neutral'
# to that grouping lands precisely on the ViT's seven classes.
#
# The MTEB task is left completely intact. It remains the
# three-way architecture comparison (TinyBERT vs DistilBERT vs
# MobileBERT); this task is what the live system deploys.
# ============================================================

GOEMOTIONS_DATASET_ID = "google-research-datasets/go_emotions"

GOEMOTIONS_CONFIG = "simplified"

# The 28 label names in the order the dataset defines them.
# Asserted against the downloaded features in data.py, for the
# same reason load_splits() asserts the MTEB order: the integer
# ids are what the grouping below is keyed on.
GOEMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
)


# ============================================================
# SHARED LABEL SPACE
# ============================================================
#
# Mirrors vit-emotion-v2-deployment/labels.py, which is the
# source of truth because the order is frozen into the exported
# ONNX classifier head. Duplicated rather than imported so this
# module stays runnable on its own; test_text_module.py asserts
# the two copies agree whenever both are importable.
# ============================================================

SHARED_LABELS = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)

SHARED_ID2LABEL = dict(enumerate(SHARED_LABELS))

SHARED_LABEL2ID = {label: index for index, label in SHARED_ID2LABEL.items()}

NUM_SHARED_LABELS = len(SHARED_LABELS)


# ============================================================
# EKMAN GROUPING
# ============================================================
#
# This is the mapping published with the GoEmotions paper
# (ekman_mapping.json in google-research/google-research), with
# its six group names renamed to the ViT spelling and 'neutral'
# added as a seventh. Using the authors own grouping rather than
# an ad-hoc one means the collapse is citable and not a
# judgement call made here.
#
# Two groupings are worth flagging in the writeup because they
# are not obvious:
#
#   'surprise' absorbs confusion, curiosity and realization -
#   Ekman treats surprise as the reaction to the unexpected,
#   which covers being puzzled by it as well as startled.
#
#   'happy' absorbs twelve of the 28 labels, everything from
#   gratitude to pride. It is by far the widest group, which is
#   part of why the text model is strongest on 'happy'.
# ============================================================

EKMAN_TO_SHARED = {
    "anger":    ("anger", "annoyance", "disapproval"),
    "disgust":  ("disgust",),
    "fear":     ("fear", "nervousness"),
    "joy":      ("admiration", "amusement", "approval", "caring", "desire",
                 "excitement", "gratitude", "joy", "love", "optimism",
                 "pride", "relief"),
    "sadness":  ("disappointment", "embarrassment", "grief", "remorse",
                 "sadness"),
    "surprise": ("confusion", "curiosity", "realization", "surprise"),
    "neutral":  ("neutral",),
}

# Ekman group name -> the ViT spelling of the same emotion.
EKMAN_GROUP_TO_SHARED = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

# Flattened: GoEmotions label name -> shared label name.
GOEMOTIONS_TO_SHARED = {
    fine_label: EKMAN_GROUP_TO_SHARED[group]
    for group, fine_labels in EKMAN_TO_SHARED.items()
    for fine_label in fine_labels
}

# GoEmotions label id -> shared label id.
GOEMOTIONS_ID_TO_SHARED_ID = {
    index: SHARED_LABEL2ID[GOEMOTIONS_TO_SHARED[name]]
    for index, name in enumerate(GOEMOTIONS_LABELS)
}


# ============================================================
# TASK REGISTRY
# ============================================================


class Task:
    """One dataset plus the label space it is trained in.

    Everything downstream - dataloaders, metrics, checkpoint
    paths, results files - is keyed on this, so the two tasks
    cannot silently share a checkpoint directory or be scored
    against each other label set.
    """

    def __init__(self, key, dataset_id, id2label, description):
        self.key = key
        self.dataset_id = dataset_id
        self.id2label = id2label
        self.label2id = {label: index for index, label in id2label.items()}
        self.num_labels = len(id2label)
        self.description = description

    @property
    def labels(self):
        return [self.id2label[index] for index in sorted(self.id2label)]

    def checkpoint_path(self, model_key):
        """Per-task checkpoint directory.

        The MTEB task keeps the original bare 'checkpoints/<model>'
        layout so the already-trained six-class checkpoints and the
        benchmark.json that describes them stay valid.
        """

        if self.key == "mteb":
            return CHECKPOINT_DIR / model_key

        return CHECKPOINT_DIR / f"{self.key}-{model_key}"

    @property
    def results_file(self):
        if self.key == "mteb":
            return RESULTS_FILE

        return RESULTS_DIR / f"benchmark_{self.key}.json"

    def __repr__(self):
        return f"Task({self.key!r}, {self.num_labels} labels)"


TASKS = {
    "mteb": Task(
        key="mteb",
        dataset_id=DATASET_ID,
        id2label=ID2LABEL,
        description="MTEB EmotionClassification, 6 classes, no neutral.",
    ),
    "goemotions": Task(
        key="goemotions",
        dataset_id=GOEMOTIONS_DATASET_ID,
        id2label=SHARED_ID2LABEL,
        description="GoEmotions collapsed to the shared 7 via Ekman grouping.",
    ),
}

DEFAULT_TASK = "goemotions"


def get_task(key):
    """Look up a task, with a useful error rather than a KeyError."""

    if key not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {key!r}. Available: {available}.")

    return TASKS[key]
