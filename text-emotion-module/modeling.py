# ============================================================
# TEXT EMOTION MODULE - MODEL CONSTRUCTION
# ============================================================
#
# One place that builds a model, so training, benchmarking and
# inference are guaranteed to construct the same architecture.
# This matters because MobileBERT needs a head adaptation that
# the other two do not - see below.
# ============================================================

import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

from config import ID2LABEL, LABEL2ID, MODELS, NUM_LABELS, get_task


# ============================================================
# THE MOBILEBERT POOLED-OUTPUT PROBLEM
# ============================================================
#
# google/mobilebert-uncased is configured with
# normalization_type="no_norm", so its NoNorm layers apply an
# elementwise affine transform without actually normalising.
# Activations therefore grow through the 24 layers, and the
# final layer emits values around 1e7:
#
#   per-layer abs max: 9.3, 20.6, ..., 28.9, 792, 63227328
#
# The backbone itself is healthy - the masked-LM head predicts
# "the capital of france is [MASK]" -> "paris" - because that
# head applies its own LayerNorm. The pooler does not, and with
# classifier_activation=False it passes the raw first-token
# state straight to the classifier.
#
# Two consequences, both measured:
#
#   1. A randomly initialised classifier produces logits around
#      1e6, so the initial cross-entropy is ~6.7e6 instead of
#      ln(6)=1.79 and training wastes most of its budget just
#      shrinking the head.
#
#   2. The first-token state is dominated by an input-INDEPENDENT
#      component. Across different sentences the raw vectors have
#      pairwise cosine similarity 0.9999, so the part that
#      identifies the sentence is ~1e-4 of the magnitude.
#
# Setting classifier_activation=True does not fix (2): the
# pretrained pooler dense times a 1e7 input saturates tanh, and
# every sentence collapses to the same +/-1 vector.
#
# LayerNorm does not fix (2) either. It centres across the
# feature dimension, but the offending component is constant
# across EXAMPLES, not across features - 408 of the 512
# dimensions exceed 1e5, so there is no small set of outliers to
# remove.
#
# BatchNorm1d does fix it, because it centres and scales each
# dimension across the batch, which is exactly the axis the
# constant component lives on. Measured on 64 training rows:
#
#   raw first-token state       pairwise cosine  mean 0.9999
#   after per-dimension centring pairwise cosine mean 0.0069
#
# So MobileBERT gets a BatchNorm1d in front of its classifier.
# Everything else - optimiser, schedule, loss, data, seed - is
# identical across all three models.
# ============================================================

NEEDS_POOLED_NORMALIZATION = {"mobilebert"}


def adapt_classifier_head(model, model_key, num_labels=None):
    """Insert a BatchNorm1d before the classifier where required."""

    if model_key not in NEEDS_POOLED_NORMALIZATION:
        return model

    # Read the head width from the loaded config when the caller did
    # not say, so reloading a checkpoint rebuilds the head it was
    # saved with rather than whichever task happens to be the default.
    if num_labels is None:
        num_labels = int(getattr(model.config, "num_labels", NUM_LABELS))

    hidden_size = model.config.hidden_size

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, num_labels),
    )

    return model


# ============================================================
# BUILD
# ============================================================

def create_model(model_key, task=None):
    """A pretrained backbone with a fresh classification head.

    task selects the label space. It defaults to the six-class MTEB
    task so the original training runs reproduce exactly.
    """

    task = get_task("mteb") if task is None else task

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_key]["hub_id"],
        num_labels=task.num_labels,
        id2label=task.id2label,
        label2id=task.label2id,
    )

    return adapt_classifier_head(model, model_key, task.num_labels)


def load_model(model_key, directory):
    """Load a fine-tuned checkpoint, rebuilding the adapted head first.

    from_pretrained cannot restore the adapted head on its own - it
    builds a plain Linear and would silently leave the saved
    classifier weights behind - so the head is rebuilt and the full
    state dict is then loaded strictly.
    """

    model = AutoModelForSequenceClassification.from_pretrained(directory)

    adapt_classifier_head(model, model_key)  # width taken from the config

    if model_key in NEEDS_POOLED_NORMALIZATION:
        model.load_state_dict(
            load_file(directory / "model.safetensors"),
            strict=True,
        )

    return model
