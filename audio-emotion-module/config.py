# ============================================================
# AUDIO EMOTION MODULE - CONFIGURATION
# ============================================================
#
# Central registry for the audio modality of the multimodal
# emotion recognition system.
#
# This module exists because the audio leg of the multimodal
# branch was never trained. What that branch shipped was:
#
#   audio_cnn.py      an untrained MFCC CNN over 5 classes
#   main.py           a fusion demo on torch.randn tensors
#   audio-module.ipynb feature extraction that ran on Kaggle,
#                     then stopped before the training cell
#
# and the corpus it extracted was both incomplete and skewed:
#
#   happy 1647   sad 1647   angry 1647   fear 1647   surprise 192
#
# Five of the shared seven classes, no neutral, no disgust, and
# an 8.6:1 imbalance on surprise. Neutral is the resting state of
# a real session - a model that cannot emit it will label silence
# as whichever class its prior favours, every frame. See
# prepare_audio_data.py for how the corpus is rebuilt.
#
# Everything here speaks the SHARED label space defined in
# ../vit-emotion-v2-deployment/labels.py, so an audio probability
# vector can be indexed against a face or text one without a
# translation step.
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODULE_DIR.parent / "vit-emotion-v2-deployment"))

from labels import SHARED_LABEL2ID, SHARED_LABELS  # noqa: E402


# ============================================================
# PATHS
# ============================================================

DATA_DIR = MODULE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

CHECKPOINT_DIR = MODULE_DIR / "checkpoints"
RESULTS_DIR = MODULE_DIR / "results"

RESULTS_FILE = RESULTS_DIR / "benchmark.json"


# ============================================================
# LABEL SPACE
# ============================================================
#
# The audio model's head emits the shared seven in shared order,
# so no projection is needed at fusion time. Contrast the text
# modality, whose MTEB checkpoints predict six classes and have
# to be bridged through TEXT_TO_SHARED_LABEL.
# ============================================================

LABELS = SHARED_LABELS
NUM_LABELS = len(LABELS)
LABEL2ID = dict(SHARED_LABEL2ID)
ID2LABEL = {index: name for name, index in LABEL2ID.items()}


# ============================================================
# SOURCE CORPORA
# ============================================================
#
# Four corpora, chosen to cover the shared seven and to span two
# very different recording conditions.
#
#   CREMA-D   7442   91 actors, 12 sentences, studio, 6 emotions
#   RAVDESS   1440   24 actors, 2 sentences, studio, 8 emotions
#   SAVEE      480    4 actors, studio, 7 emotions
#   MELD     13708   Friends dialogue, in-the-wild, 7 emotions
#
# Why all four rather than the largest one
# ----------------------------------------
# CREMA-D alone is the obvious pick on volume, but it has no
# surprise class at all - which is precisely the hole the old
# corpus had. RAVDESS and SAVEE are the only acted corpora here
# that record surprise, and between them they contribute 252
# clips of it. That is not enough on its own either.
#
# MELD closes the gap. It is television dialogue rather than
# acted prompts: 1205 surprise clips in its train split, with
# laugh track, music, overlapping speakers and clips as short as
# a quarter of a second. It is much harder than the studio
# corpora - audio-only MELD sits near 50% weighted F1 in the
# literature, against ~75% for CREMA-D - so mixing it in costs
# some accuracy on the studio domain. It buys two things worth
# more than that: surprise coverage, and a model that has heard
# what a laptop microphone in a living room actually sounds like.
#
# The two domains are tracked separately end to end (the
# `domain` column of the manifest) and reported separately, so
# the studio number is never quietly passed off as the
# in-the-wild one.
#
# Why MELD matters a second time
# ------------------------------
# MELD ships a transcript per clip. That makes its test split the
# only set here where the audio and the text modality describe
# the same utterance, so it is the one place the fusion can be
# measured on real pairs instead of the synthetic pairing that
# eval_fusion.py had to use. See eval_multimodal.py.
# ============================================================

SOURCES = {
    # --------------------------------------------------------
    # CREMA-D: 91 actors x 12 sentences x 6 emotions.
    # Filenames are ACTOR_SENTENCE_EMOTION_INTENSITY.wav, e.g.
    # 1068_TIE_ANG_XX.wav, so the speaker id is the leading field.
    # --------------------------------------------------------
    "cremad": {
        "hub_id": "confit/cremad-parquet",
        "configs": {"default": ("train", "validation", "test")},
        "domain": "acted",
        "sampling_rate": 16000,
        # The dataset's own ClassLabel order. Asserted on load so an
        # upstream reordering cannot silently relabel the corpus.
        "names": ("anger", "disgust", "fear", "happy", "neutral", "sad"),
        "map": {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
        },
    },
    # --------------------------------------------------------
    # RAVDESS: the parquet mirror carries speech AND song, 1440
    # clips each, and partitions them into five speaker folds.
    # fold1's train+test is the whole 2880, so one config is
    # enough; the fold boundaries are ignored in favour of this
    # module's own speaker split.
    #
    # Filenames are seven dash-separated fields:
    #   03-01-02-01-02-02-06.wav
    #    |  |  |  |  |  |  +- actor      01..24
    #    |  |  |  |  |  +---- repetition
    #    |  |  |  |  +------- statement
    #    |  |  |  +---------- intensity  01 normal, 02 strong
    #    |  |  +------------- emotion    01..08
    #    |  +---------------- channel    01 speech, 02 song
    #    +------------------- modality   03 audio-only
    # --------------------------------------------------------
    "ravdess": {
        "hub_id": "confit/ravdess-parquet",
        "configs": {"fold1": ("train", "test")},
        "domain": "acted",
        "sampling_rate": 48000,
        "names": (
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ),
        "map": {
            "neutral": "neutral",
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "fearful": "fear",
            "disgust": "disgust",
            "surprised": "surprise",
            # "calm" is deliberately absent - see DROP_RAVDESS_CALM.
        },
    },
    # --------------------------------------------------------
    # SAVEE: 4 male actors, filenames SPEAKER_CODE##.wav where
    # the code is a/d/f/h/n/sa/su. The hub copy already carries
    # an `emotion` string column, so the code is only used for
    # the speaker id.
    # --------------------------------------------------------
    "savee": {
        "hub_id": "AbstractTTS/SAVEE",
        "configs": {"default": ("train",)},
        "domain": "acted",
        "sampling_rate": None,  # resampled on decode
        "names": (
            "anger",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise",
        ),
        "map": {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "happiness": "happy",
            "neutral": "neutral",
            "sadness": "sad",
            "surprise": "surprise",
        },
    },
    # --------------------------------------------------------
    # MELD: fetched from its tar archives plus the per-split CSVs
    # rather than through `datasets`, because the hub copy is a
    # loading script and the datasets library no longer runs
    # those. prepare_audio_data.py streams the tars directly.
    # --------------------------------------------------------
    "meld": {
        "hub_id": "ajyy/MELD_audio",
        "configs": None,  # handled by the tar reader, not `datasets`
        "domain": "wild",
        "sampling_rate": None,
        "names": (
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ),
        "map": {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "joy": "happy",
            "neutral": "neutral",
            "sadness": "sad",
            "surprise": "surprise",
        },
    },
}

MELD_SPLIT_FILES = {
    "train": ("train.csv", "archive/train.tar.gz"),
    "validation": ("dev.csv", "archive/dev.tar.gz"),
    "test": ("test.csv", "archive/test.tar.gz"),
}


# ============================================================
# RAVDESS "calm"
# ============================================================
#
# RAVDESS records 376 clips of a "calm" emotion that no other
# corpus here has and that the shared space has no slot for. The
# usual move in the SER literature is to fold it into neutral.
#
# It is dropped instead. Calm is a *deliberately* serene read -
# lower pitch variance and slower rate than this corpus's
# neutral, which is itself an unemotional read of the same two
# sentences. Folding them together teaches the model that one
# prosodic pattern maps to two quite different acoustic clusters,
# and neutral is the class the fusion leans on hardest: it is
# what should win when nobody is expressing anything, and a
# smeared neutral shows up as a jittery idle state in the live
# session. CREMA-D and SAVEE already supply 1207 clean neutral
# clips, so the 376 are not needed for volume.
#
# Set to False to include them as neutral.
# ============================================================

DROP_RAVDESS_CALM = True


# ============================================================
# RAVDESS song
# ============================================================
#
# The parquet mirror bundles the sung half of RAVDESS. Sung
# emotion has prosody driven by melody rather than by affect, and
# the deployment target is someone talking at a laptop. The song
# half also records neither disgust nor surprise, so excluding it
# costs nothing on the two classes that are actually short.
# ============================================================

INCLUDE_RAVDESS_SONG = False


# ============================================================
# AUDIO PREPROCESSING
# ============================================================
#
# 16 kHz mono. That is what every pretrained speech encoder worth
# using expects, and emotional prosody lives far below the 8 kHz
# Nyquist limit - pitch, energy contour and speaking rate are all
# sub-1 kHz phenomena.
#
# CLIP_SECONDS = 4.0 covers the 95th percentile of the acted
# corpora without padding most clips to twice their length.
# Longer clips are cropped to their most energetic window rather
# than their first 4 seconds: CREMA-D and RAVDESS both open with
# a few hundred milliseconds of room tone, and a fixed head crop
# spends a tenth of the input on it.
# ============================================================

SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)


# ============================================================
# QUALITY FILTERS
# ============================================================
#
# MELD is cut from episode audio by subtitle timestamps, so a
# fraction of its clips are pure laugh track, pure music, or a
# single frame of a slammed door with no speech in them at all.
# Those rows carry an emotion label that nothing in the audio
# supports, and they are label noise of exactly the kind the face
# module had to clean out of FER2013 (see training/clean_labels.py).
#
#   MIN_VOICED_SECONDS  a clip needs this much above-floor audio
#                       to be worth a label at all
#   MIN_RMS             drops digital silence and near-silence
#   SILENCE_TOP_DB      librosa.effects.trim threshold, in dB
#                       below the clip's own peak
# ============================================================

MIN_VOICED_SECONDS = 0.40
MIN_RMS = 0.0025
SILENCE_TOP_DB = 30.0


# ============================================================
# SPLITS
# ============================================================
#
# The acted corpora are split by SPEAKER, never by clip.
#
# This is the single most important decision in this file. Every
# acted corpus here has each actor perform every emotion on the
# same handful of sentences, so a random clip split puts the same
# voice saying the same words in both train and test. A model can
# then score well by recognising the actor and recalling which of
# their takes was which - and measured on CREMA-D, a random split
# reads 15-25 points above a speaker-disjoint one. The inflated
# number is the one most SER papers quote.
#
# MELD keeps its own official dialogue-level split, which is how
# every MELD number in the literature is produced.
#
# Proportions are approximate: speakers are whole and unequal, so
# the allocator fills test, then validation, then gives the rest
# to train.
# ============================================================

VAL_FRACTION = 0.12
TEST_FRACTION = 0.16

SEED = 42


# ============================================================
# BALANCING
# ============================================================
#
# Two separate levers, because imbalance hurts in two places.
#
# CLASS_CAP_PERCENTILE caps the TRAIN split only. After assembly
# neutral is roughly 6k and fear roughly 2.6k; left alone the
# model spends most of its gradient on neutral and learns to
# answer "neutral" whenever it is unsure. Each class is capped at
# this percentile of the class-count distribution, subsampling the
# over-full ones proportionally across their source corpora so
# that capping neutral does not quietly delete all of MELD's
# neutral and keep all of CREMA-D's.
#
# Nothing is oversampled and nothing is synthesised. Duplicating
# the 1.7k surprise clips to match neutral would not add
# information, it would just raise the learning rate on those
# particular clips - and the residual imbalance is better handled
# in the loss, where it can be tuned without touching the data.
# See CLASS_WEIGHT_POWER in train_audio.py.
#
# Validation and test are left at their natural distribution. A
# balanced test set would make accuracy and macro-F1 agree by
# construction and hide exactly the failure this module is trying
# to avoid.
# ============================================================

CLASS_CAP_PERCENTILE = 70.0


# ============================================================
# MODEL REGISTRY
# ============================================================
#
# Two models, for the same reason text-emotion-module benchmarks
# three: the comparison is the result.
#
# mfcc-cnn is the architecture the multimodal branch already had,
# retrained unchanged on the rebuilt corpus. It isolates how much
# of the audio leg's problem was the data and how much was the
# model. Without it, a good number from the pretrained encoder
# proves only that the pair of changes worked together.
#
# wavlm-base-plus is the model meant for deployment. Self-
# supervised speech encoders dominate emotion recognition for the
# same reason ViT-base beat a scratch CNN on faces: 94k hours of
# unlabelled speech teaches a representation that 16k labelled
# clips cannot.
#
# Why WavLM rather than wav2vec2 or HuBERT
# ----------------------------------------
# All three share an architecture and differ in pretraining.
# WavLM adds overlapped-speech denoising and simulated noise to
# the masked-prediction objective, and it leads base-size models
# on SUPERB's emotion-recognition task. The denoising pretext is
# the relevant part here: a third of this corpus is MELD, which
# is dialogue under a laugh track with speakers talking over each
# other, and that is the condition WavLM was pretrained to handle.
#
# Why the feature extractor is frozen
# -----------------------------------
# The convolutional front end maps raw 16 kHz samples to 50 Hz
# frames and is shared by every downstream task. Fine-tuning it
# on 16k clips destabilises training - gradients reach it through
# the whole transformer - for no measured gain. This is the
# standard recipe, not a shortcut.
#
# Why a weighted sum of all hidden layers
# ---------------------------------------
# Emotion is not a last-layer property. In a masked-prediction
# encoder the top layers specialise towards the pretext task
# (predicting the masked unit, which is close to phonetic
# identity), while paralinguistic information - pitch contour,
# energy, voice quality - peaks in the middle. Taking only the
# last hidden state throws that away. A learned softmax over the
# 13 layer outputs lets the model choose, and it is what the
# SUPERB baselines do.
#
# Why attentive statistics pooling
# --------------------------------
# A clip has one label but ~200 frames, so the frames must be
# collapsed. Mean pooling gives the room tone at the start of a
# CREMA-D clip the same vote as the stressed vowel that carries
# the anger. Attentive pooling learns per-frame weights, and
# concatenating the weighted standard deviation to the weighted
# mean keeps the variability - how much the prosody MOVES across
# the clip, which is most of what separates an excited read from
# a flat one.
# ============================================================

MODELS = {
    "mfcc-cnn": {
        "kind": "mfcc-cnn",
        "display_name": "MFCC CNN (4 blocks)",
        "hub_id": None,
        "note": (
            "The multimodal branch's own AudioCNN, unchanged, retrained "
            "on the rebuilt corpus. Baseline."
        ),
        "learning_rate": 1e-3,
        "backbone_learning_rate": None,
        "batch_size": 64,
        "epochs": 30,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.05,
        "patience": 8,
    },
    "wavlm-base-plus": {
        "kind": "ssl",
        "display_name": "WavLM-base+ (12L-768D)",
        "hub_id": "microsoft/wavlm-base-plus",
        "note": (
            "Self-supervised speech encoder, 94k hours. Weighted layer "
            "sum into attentive statistics pooling."
        ),
        # The head is new and the backbone is not: one learning rate
        # for both either moves the head too slowly to converge in a
        # handful of epochs or moves the backbone fast enough to erase
        # the pretraining. Two rates, 20x apart.
        "learning_rate": 1e-3,
        "backbone_learning_rate": 5e-5,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "epochs": 6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "patience": 3,
        "mask_time_prob": 0.05,
        # ----------------------------------------------------
        # LayerDrop is OFF, and has to be.
        #
        # The usual SSL fine-tuning recipe sets it to 0.05. It is
        # incompatible with the weighted layer sum above. When
        # transformers drops a layer it also drops that layer's entry
        # from output_hidden_states, so the tuple arrives 11, 12 or 13
        # long depending on the draw. The short cases raise outright;
        # the subtler damage is that the survivors shift down a slot,
        # so the mixture weight trained for layer 7 lands on layer 8
        # and the one thing the layer sum is supposed to learn - WHERE
        # in the stack emotion lives - is scrambled every step.
        #
        # SpecAugment (mask_time_prob) stays on and covers the
        # regularisation LayerDrop would have provided.
        # ----------------------------------------------------
        "layerdrop": 0.0,
    },
}

DEFAULT_MODEL = "wavlm-base-plus"


# ============================================================
# TRAINING DEFAULTS
# ============================================================
#
# CLASS_WEIGHT_POWER controls how hard the loss compensates for
# the imbalance the train-split cap leaves behind:
#
#   weight[c] = (N / count[c]) ** CLASS_WEIGHT_POWER
#
#   0.0   no compensation; the model answers with the prior and
#         macro-F1 collapses on the rare classes
#   1.0   full inverse frequency; the rare classes dominate the
#         gradient, the model over-predicts them, and precision
#         on them falls faster than recall rises
#   0.5   inverse square root
#
# 0.5 is the setting here. It is the standard compromise, and the
# cap has already removed the worst of the skew, so the loss only
# has to cover a ~2:1 residual rather than the ~6:1 the raw
# corpus carries.
#
# LABEL_SMOOTHING matters more than usual for this task. Acted
# emotion labels are the actor's intent, not a measurement: a
# CREMA-D "fear" take that reads as surprise is still filed under
# fear. Smoothing stops the model from driving those to a
# confident wrong answer, and a modality that is wrong but
# uncertain costs the fusion far less than one that is wrong and
# certain.
# ============================================================

SEED_TRAIN = 42

CLASS_WEIGHT_POWER = 0.5
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0


# ============================================================
# MFCC FEATURES (baseline model only)
# ============================================================
#
# Matches the multimodal branch's extract_mfcc so the baseline is
# genuinely that branch's model and not a redesign of it: 40
# coefficients, hop 512, per-coefficient normalisation. The only
# change is the frame count, which follows CLIP_SECONDS here
# instead of being hard-coded to 130.
# ============================================================

N_MFCC = 40
MFCC_HOP_LENGTH = 512
MFCC_FRAMES = 1 + CLIP_SAMPLES // MFCC_HOP_LENGTH


# ============================================================
# HELPERS
# ============================================================

def checkpoint_path(model_key: str) -> Path:
    """Directory holding the fine-tuned weights for one model."""

    return CHECKPOINT_DIR / model_key


def ensure_directories() -> None:
    for directory in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
