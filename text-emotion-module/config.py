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
