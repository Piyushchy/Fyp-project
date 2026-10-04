# Audio Emotion Module

The voice leg of the multimodal emotion recogniser. Takes a 4-second
window of 16 kHz mono audio and returns a probability over the same
seven emotions the face and text modules use, so the three can be
pooled without a translation step.

```
angry   disgust   fear   happy   neutral   sad   surprise
```

---

## Why this module exists

The `multimodal-upgrade` branch introduced an audio leg but never
trained it. What that branch actually contained:

| File | State |
|---|---|
| `audio_cnn.py` | An `AudioCNN` over **5** classes. Architecture only — no trained weights. |
| `main.py` | A `MultimodalTransformerFusion` demo whose inputs are `torch.randn(batch, 1, 40, 130)` and `torch.randn(batch, 768)`. It fuses noise. |
| `audio-module.ipynb` | MFCC extraction that ran on Kaggle and stopped at the feature-saving cell, before any training cell. |

And the corpus it extracted was both incomplete and skewed:

```
happy 1647   sad 1647   angry 1647   fear 1647   surprise 192
```

Five of the shared seven. **No neutral and no disgust**, and an 8.6:1
imbalance on surprise. The missing neutral is the serious one: neutral
is the resting state of a real session, and a model that cannot emit
it labels every silent frame as whichever class its prior favours.

This module rebuilds the corpus and trains the leg.

### What changed

| | `multimodal-upgrade` | here |
|---|---|---|
| Classes | 5 (no neutral, no disgust) | 7, the shared space |
| Clips | 6,780 | 20,012 |
| Sources | CREMA-D, RAVDESS | CREMA-D, RAVDESS, SAVEE, MELD |
| Worst imbalance | 8.6 : 1 | 1.65 : 1 |
| Split | random over clips | **speaker-disjoint**, MELD on its official split |
| Domains | studio only | studio + in-the-wild, reported separately |
| Trained | no | yes |
| Fusion input | `torch.randn` | real probabilities, pooled by `fusion.py` |

---

## The corpus

Built by `prepare_audio_data.py` from four public corpora:

| Source | Clips kept | Domain | Why it is here |
|---|---|---|---|
| CREMA-D | 7,441 | acted | Volume. 91 actors, 6 emotions, no surprise. |
| RAVDESS | 2,362 | acted | One of only two sources here recording surprise. |
| SAVEE | 480 | acted | Covers all seven, including surprise. |
| MELD | 13,313 | wild | Surprise coverage, realistic audio, **and transcripts**. |

After quality filtering and class capping: **20,012 clips**.

```
class            train  validation        test   total
----------------------------------------------------------
angry             2322         349         603    3274
disgust           1485         223         341    2049
fear              1491         238         323    2052
happy             2443         361         657    3461
neutral           2443         632        1449    4524
sad               1854         307         473    2634
surprise          1494         191         333    2018
----------------------------------------------------------
total            13532        2301        4179   20012
```

### Three decisions worth defending

**Split by speaker, never by clip.** Every acted corpus here has each
actor perform every emotion on the same handful of sentences. A random
clip split puts the same voice saying the same words in both train and
test, so a model can score well by recognising the actor and recalling
which take was which. Measured on CREMA-D, a random split reads 15–25
points above a speaker-disjoint one — and the inflated number is the
one most SER papers quote. 119 acted speakers here, none spanning two
splits, asserted at build time.

**Cap the train split, leave validation and test alone.** Neutral
arrives at ~5.5k against fear's ~1.5k. Capping each class at the 70th
percentile of the class-count distribution brings the worst ratio to
1.65:1, subsampling proportionally across source corpora so that
capping neutral does not quietly delete all of MELD's and keep all of
CREMA-D's. Nothing is oversampled and nothing is synthesised — the
residual imbalance is handled in the loss, where it can be tuned
without touching the data. Validation and test keep their natural
distribution, because a balanced test set makes accuracy and macro-F1
agree by construction and hides exactly the failure this module is
trying to avoid.

**Drop RAVDESS "calm".** 376 clips of an emotion the shared space has
no slot for. The usual move is to fold it into neutral; it is dropped
instead, because calm is a *deliberately* serene read with lower pitch
variance than this corpus's neutral, and merging them teaches the model
that one prosodic pattern maps to two different acoustic clusters.
Neutral is the class the fusion leans on hardest, and a smeared neutral
shows up as a jittery idle state in the live session.

### Quality filtering

MELD is cut from episode audio by subtitle timestamps, so some clips
are pure laugh track, pure music, or a single slammed door — carrying
an emotion label nothing in the audio supports. Clips need 0.40 s of
above-floor voiced audio and RMS above 0.0025 to survive. 528 clips
were dropped across all sources.

---

## Models

Two, because the comparison is the result.

| Model | Params | Role |
|---|---|---|
| `mfcc-cnn` | 391 K | The `multimodal-upgrade` branch's own `AudioCNN`, imported unchanged from `../audio_cnn.py`, retrained on the rebuilt corpus. |
| `wavlm-base-plus` | 95.0 M | WavLM-base+ with a weighted layer sum into attentive statistics pooling. The deployment model. |

The baseline is not a straw man — it is literally that branch's
architecture, imported rather than reimplemented, with `num_classes`
moved 5 → 7. Without it, a good number from WavLM would not
distinguish "the data was the problem" from "the model was the
problem".

### WavLM head

- **Frozen feature encoder.** The convolutional front end maps raw
  samples to 50 Hz frames and is task-agnostic; fine-tuning it through
  the whole transformer destabilises training for no measured gain.
- **Weighted layer sum.** Emotion is not a last-layer property. In a
  masked-prediction encoder the top layers specialise towards phonetic
  identity while pitch, energy and voice quality peak in the middle. A
  learned softmax over the 13 layer outputs lets the model choose, and
  `train_audio.py` prints where it landed.
- **Attentive statistics pooling.** A clip has one label but ~200
  frames. Mean pooling gives the room tone at the start of a CREMA-D
  clip the same vote as the stressed vowel carrying the anger.
  Attentive pooling learns per-frame weights, and concatenating the
  weighted standard deviation keeps how much the prosody *moves*.

### LayerDrop is off, and has to be

The standard SSL recipe sets `layerdrop=0.05`. It is incompatible with
a weighted layer sum: when `transformers` drops a layer it also drops
that layer's entry from `output_hidden_states`, so the tuple arrives
11, 12 or 13 long depending on the draw. The short cases raise
outright; the subtler damage is that the survivors shift down a slot,
so the weight trained for layer 7 lands on layer 8 and the one thing
the layer sum exists to learn is scrambled every step. `modeling.py`
now raises rather than letting a length mismatch through silently.

---

## Results

Speaker-disjoint test split, 4,179 clips, seven classes (chance is
14.3%).

| Model | Params | Accuracy | Macro-F1 | Weighted-F1 | ECE | ms/clip |
|---|---|---|---|---|---|---|
| MFCC CNN (the branch's model) | 391 K | 0.3147 | 0.3218 | 0.2875 | 0.137 | 0.6 |
| **WavLM-base+** | 95.0 M | **0.4901** | **0.4988** | **0.4946** | 0.148 | 16.5 |

**+17.7 macro-F1 points** from replacing the architecture, on identical
data. Both numbers come from the same corpus, the same splits and the
same loss, so the gap is the model and nothing else.

### The domain split is the real story

| Model | Domain | Accuracy | Macro-F1 | n |
|---|---|---|---|---|
| MFCC CNN | acted | 0.4529 | 0.4119 | 1,656 |
| MFCC CNN | wild (MELD) | 0.2239 | 0.1555 | 2,523 |
| WavLM | acted | **0.6938** | **0.6906** | 1,656 |
| WavLM | wild (MELD) | 0.3563 | 0.2464 | 2,523 |

On studio audio WavLM reaches 69.4% accuracy, which is where the
literature puts base-size SSL encoders on CREMA-D with speaker-disjoint
splits. On MELD it reaches 35.6%, and `disgust` (F1 0.027) and `fear`
(F1 0.049) essentially do not work — both have under 70 test clips and
both are the classes MELD annotates from dialogue context rather than
from anything audible. **A single averaged number would have hidden
all of this**, which is why the two domains are tracked separately from
the manifest onward and why `reliability.json` is published from the
wild figures: the deployment input is a laptop microphone in a room,
not a recording booth.

### Where the encoder found the emotion

The learned softmax over the 13 layer outputs, after training:

```
L0:0.064  L1:0.061  L2:0.061  L3:0.061  L4:0.063  L5:0.065  L6:0.069
L7:0.079  L8:0.102  L9:0.108  L10:0.099  L11:0.089  L12:0.079
```

It peaks at **layers 8–10**, not at the top of the stack and not at the
embeddings. That is the weighted layer sum earning its place: reading
only the last hidden state (0.079) would have discarded the layers the
model actually relies on. Initialised uniform at 1/13 = 0.077, so every
deviation here is learned.

---

## Multimodal fusion

`eval_multimodal.py` scores the three-modality pool on the 2,523 MELD
test clips that carry both audio and a transcript — the only set in
this project where two real modalities describe the same utterance.
Audio and text are real model outputs; **face is simulated** from the
deployed model's measured confusion matrix, because no corpus here has
a face aligned to a MELD utterance.

| Modalities | Accuracy | Macro-F1 | Weighted-F1 | ECE |
|---|---|---|---|---|
| face | 0.5220 | 0.4209 | 0.5429 | 0.183 |
| audio | 0.3646 | 0.2337 | 0.3619 | 0.035 |
| text | 0.3270 | 0.2339 | 0.3105 | 0.114 |
| face + audio | 0.5363 | 0.4329 | 0.5494 | 0.159 |
| face + text | 0.5355 | 0.4276 | 0.5469 | 0.183 |
| audio + text | 0.3813 | 0.2699 | 0.3773 | 0.047 |
| **face + audio + text** | **0.5545** | **0.4431** | **0.5590** | 0.156 |

Every modality added improves the pool, and the trimodal result beats
the best single modality by **+2.2 macro-F1** and **+3.3 accuracy**.
That gain is small, and it is small for an honest reason: these are
real pairs. A quiet, flat delivery and a short, affectless sentence
co-occur, so the modalities fail together more often than an
independent pairing would suggest. `eval_fusion.py`, which has to pair
face and text synthetically, reports a much larger gain for exactly
that reason.

Adding audio also *improves calibration* — audio is the best-calibrated
leg here (ECE 0.035), and the pooled ECE falls from 0.183 (face alone)
to 0.156.

### Score

```
MULTIMODAL SCORE: 49.7 / 100

  capability               0.633 x 0.35   trimodal macro-F1 against a 0.70 ceiling
  fusion_gain              0.222 x 0.25   macro-F1 over best unimodal, against +0.10
  calibration              0.376 x 0.15   1 - ECE/0.25
  robustness               0.609 x 0.15   worst two-modality macro-F1 / trimodal
  conflict_discrimination  0.721 x 0.10   agreed minus conflicted accuracy, against 0.30
```

The composite is this project's own summary of "is the pool worth
having", not a standard metric — so every component, its raw value, its
normalisation and its weight are written to `results/multimodal.json`
alongside the total.

The component worth reading on its own is **conflict discrimination**.
When the three modalities agree the pool is right 74.1% of the time;
when the flag fires it is right 52.5%. A 21.6-point gap means the flag
predicts something real, which is what justifies surfacing it in the UI
rather than hiding the disagreement.

Fusion overhead is 0.13 ms per frame.

### Degradation

All three sweeps are monotonic in the right direction — each modality
is carrying weight, and removing it costs:

| Axis | healthy | degraded | collapse |
|---|---|---|---|
| face quality 1.0 → 0.0 | 0.4359 | 0.4143 (0.4) | 0.2699 (audio+text) |
| voiced scale 1.0 → 0.0 | 0.4431 | 0.4314 (0.4) | 0.4276 (face+text) |
| text age 0s → 200s | 0.4397 | 0.4232 (60s) | 0.4209 (face only) |

### A caveat on the tuned weights

`--tune` searches the three base weights on the MELD **validation**
split and never on test. It found `face 1.44, audio 0.48, text 0.34`,
lifting validation macro-F1 from 0.357 to 0.445.

Those weights are **not** written back to `reliability.json`. They are
tuned for MELD — television dialogue with subtitle-derived transcripts
— and text is down-weighted hard there because TinyBERT was trained on
GoEmotions (Reddit comments) and transfers poorly to it. The deployed
system's text input is a message someone actually typed, which is much
closer to GoEmotions than to a sitcom line. Publishing MELD-tuned
weights would optimise the live session for the wrong distribution.

---

## Usage

```bash
# 1. Build the corpus. Downloads ~1.5 GB of MELD archives plus the
#    HuggingFace parquet mirrors, and writes ~2.9 GB of clips and
#    data/manifest.csv. Resumable: a dropped MELD download picks up
#    from the .part file rather than starting over.
python prepare_audio_data.py --keep-meld-archives

# 2. Train
python train_audio.py --model mfcc-cnn
python train_audio.py --model wavlm-base-plus

# 3. Compare, time, and publish the profile to the fusion engine
python benchmark_audio.py

# 4. Score the three-modality pool
python eval_multimodal.py --tune
```

The repository ships `data/manifest.csv`, not the waveforms. The
manifest is the corpus specification — one row per clip with its uid,
label, speaker, split, domain, transcript and measured quality — and
`prepare_audio_data.py` rebuilds every clip bit-for-bit from the public
HuggingFace copies. This matches how the face corpus is handled.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Every tunable, with the reasoning for each. |
| `prepare_audio_data.py` | Downloads, decodes, filters, splits, caps, writes the manifest. |
| `data.py` | Manifest loading, augmentation, MFCC cache, loaders. |
| `modeling.py` | `WeightedLayerSum`, `AttentiveStatsPooling`, both classifiers. |
| `metrics.py` | Accuracy, macro/weighted F1, balanced accuracy, per-class F1, ECE. |
| `train_audio.py` | Training loop, early stopping on validation macro-F1. |
| `benchmark_audio.py` | Model comparison, latency, publishes to `reliability.json`. |
| `eval_multimodal.py` | Three-modality fusion scoring on MELD pairs. |
