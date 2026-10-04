# Multimodal Emotion Recognition

Real-time emotion recognition from **face and text together**. A Vision
Transformer reads facial expression from a webcam, a fine-tuned compact
transformer reads typed messages, and a reliability-weighted pool fuses
them into a single answer that explains how it was reached.

```
webcam ──▶ YuNet detect ──▶ adaptive crop ──▶ ViT ONNX ──▶ smooth ──┐
                                                                     ├──▶ fused emotion
chat  ──▶ TinyBERT (GoEmotions) ──────────────▶ decay with age ──────┘
```

Both modalities predict over the **same seven classes**, so their
probability vectors are directly comparable:

```
angry · disgust · fear · happy · neutral · sad · surprise
```

---

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows

cd vit-emotion-v2-deployment
pip install -r requirements.txt   # curated; root requirements.txt is a full freeze
python setup_models.py            # one-time: YuNet face detector (227 KB)
uvicorn server:app --port 8000
```

Open **http://localhost:8000/**.

Serve the page from the server rather than opening `index.html` off
disk — the client derives the WebSocket address from `window.location`,
and a `file://` URL has no host to derive it from.

The text modality needs a trained checkpoint (~26 min on CPU):

```bash
cd text-emotion-module
python train.py --task goemotions --model tinybert
```

Without one the server still runs face-only and the UI says so.

### GPU (optional, ~7x faster)

```bash
pip install onnxruntime-gpu
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Both modalities use it automatically and fall back to CPU silently when
it is absent; `GET /health` reports which. Measured on an RTX 4060, ViT
batch-of-8: **63 ms CPU → 8.8 ms CUDA**. `EMOTION_DEVICE=cpu` forces CPU.

---

## Repository layout

| Directory | What is in it |
|---|---|
| [`vit-emotion-v2-deployment/`](vit-emotion-v2-deployment/README.md) | The running system: face pipeline, fusion, FastAPI server, web UI, evaluation |
| [`text-emotion-module/`](text-emotion-module/README.md) | The text modality: two training tasks, three architectures compared, benchmarks |
| [`audio-emotion-module/`](audio-emotion-module/README.md) | The audio modality: four-corpus rebuild, speaker-disjoint splits, WavLM vs the branch's MFCC CNN |
| [`training/`](training/) | Multi-dataset retraining for the face ViT: local GPU script + Colab notebook |

Each directory has its own README with the detail. This page is the map.

---

## The face modality

A ViT fine-tuned for seven-class facial expression, exported to ONNX.

**Measured on FER2013's test split: 74.40% accuracy / 73.72%
macro-F1.** The project README previously claimed 67.69%; running the
model directly shows it is better than that.

The problem is not the in-dataset score. It is that the same weights
score **51.80%** on RAF-DB — faces the model has never seen the like
of. That 23-point gap is what a webcam user actually experiences, and
it is the engineering problem this work attacks.

### What the inference-time changes are actually worth

> **Corrected.** An earlier revision of this README claimed eye-line
> alignment was worth about +4 macro-F1. That was measured on
> composited out-of-distribution images where both configurations
> scored badly, so the comparison was meaningless. Measured on the
> model's own distribution, alignment is a net loss on upright faces.

First, a sanity check the original work skipped: run the ONNX model on
FER2013 with a plain resize and no detector at all.

**74.50% accuracy / 72.83% macro-F1** — *better* than the 67.69% this
README used to claim. The weights and the preprocessing config were
never the problem.

Then, varying only the crop:

| configuration | FER2013 accuracy | macro-F1 |
|---|---|---|
| raw resize, no detector | **71.93%** | **70.63%** |
| detector + box crop | 70.76% | 69.19% |
| detector + aligned warp (best of 40 geometries) | 66.55% | 63.85% |

Alignment loses everywhere. It resamples an already small crop and
replicates edge pixels; on an upright face there is no pose error to
win back.

It inverts under roll, though — aligned minus box crop:

| roll | 0° | 10° | 15° | 20° | 25° | 30° |
|---|---|---|---|---|---|---|
| delta | −2.8 | −3.3 | −3.1 | **+4.2** | **+5.5** | **+8.7** |

So the deployed default is **adaptive**: box crop below ~18° of roll,
aligned warp above it. Pay for the warp only where it buys something.

Also fixed along the way: the original `server.py` branched on
`hasattr(mp, "solutions")` to choose MediaPipe over Haar. On this
environment mediapipe 0.10.35 under Python 3.14 ships only
`mediapipe.tasks`, so that test was always false and **every request
silently took the Haar path** — which is why `mediapipe_confidence`
always rendered as `—`.

### The actual problem: cross-dataset generalisation

| dataset | accuracy | |
|---|---|---|
| FER2013 test | **74.40%** | the model's own distribution |
| RAF-DB | **51.80%** | never seen in training |

A 23-point drop on unseen faces, and no crop tuning touches it. This is
what a webcam user experiences, and it needs retraining.

### Retraining on three datasets (`training/`)

```bash
cd training
python prepare_face_data.py      # merges the corpus, ~15 min
python train_face.py             # ~30-50 min on an RTX 4060
```

Same architecture, same seven-class head, same ONNX signature — only
the data changes:

| source | images | why it is there |
|---|---|---|
| FER2013 train | 28,709 | volume, and the distribution the current model knows |
| AffectNet | 27,823 | real photographs, better labels |
| RAF-DB | 20,471 | real photographs, crowd-annotated |
| **total** | **77,003** | |

Plus class-balanced loss and webcam-domain augmentation — JPEG
artefacts, motion blur, uneven exposure, occlusion, grayscale.

**Result** (3 epochs, ~30 min on an RTX 4060), accuracy / macro-F1:

| test set | v2 (FER2013 only) | v3 (3 datasets) | delta |
|---|---|---|---|
| FER2013 test — v2's home turf | 72.90% / 72.77% | 69.04% / 66.20% | −3.9 / −6.6 |
| **A 4th dataset, unseen by both** | 40.57% / 36.74% | **48.00% / 46.61%** | **+7.4 / +9.9** |

**v3 is worse on FER2013 and better on everything else** — which is the
trade, stated in both directions. v2 was trained on FER2013, so FER2013
flatters it. The second row is the one that describes a stranger at a
webcam, and on unseen faces `disgust` goes from 5.5 to 28.8 F1 and
`sad` from 38.7 to 56.7.

v3 is the deployed default; `EMOTION_MODEL=v2` switches back.

`train_face_vit.ipynb` is the Colab/Kaggle version for machines without
a local GPU.

---

## The text modality

Three compact transformers fine-tuned and compared, on two tasks.

| Model | Params | MTEB (6-class) acc | Latency |
|---|---|---|---|
| TinyBERT (4L-312D) | 14.4M | 92.30% | 2.8 ms |
| DistilBERT (6L-768D) | 67.0M | 93.15% | 10.6 ms |
| MobileBERT (24L-512D) | 24.6M | 93.25% | 16.9 ms |

BERT-base and RoBERTa are deliberately excluded — the visual module
already committed the project to real-time CPU inference.

**But the six-class MTEB label space could not be deployed.** It has no
`neutral` and no `disgust`. Since most typed messages in a live session
are affectively neutral, a six-class model asserts an emotion on every
one of them, having no way to say "nothing in particular" — and two of
the seven fused classes could only ever be argued for by one modality.

So a second task was added: **GoEmotions**, collapsed onto the seven
shared classes using the grouping published with the GoEmotions paper.

**TinyBERT/GoEmotions: 65.64% accuracy / 59.52% macro-F1**, over all
seven classes.

That is far below the 92.30% on MTEB, and the drop is expected rather
than a regression. MTEB emotion is first-person *"i feel X"* tweets —
near-template phrasing where the label is often stated outright.
GoEmotions is real Reddit comments, seven classes, with `neutral`
covering a third of the data. **The MTEB figure was optimistic for the
deployment setting; the GoEmotions figure is the one that reflects what
the live system does.**

---

## The audio modality

The `multimodal-upgrade` branch introduced an audio leg but never
trained it: `audio_cnn.py` held an untrained `AudioCNN` over five
classes, `main.py` fused `torch.randn` tensors, and the Kaggle notebook
stopped before any training cell. The corpus it had extracted was five
of the seven classes — **no neutral, no disgust** — at 8.6:1 imbalance.

[`audio-emotion-module/`](audio-emotion-module/README.md) rebuilds the
corpus from four sources (CREMA-D, RAVDESS, SAVEE, MELD) into 20,012
clips covering all seven classes at 1.65:1, **split by speaker rather
than by clip**. That last point matters more than any model choice:
every acted corpus has each actor perform every emotion on the same
sentences, so a random clip split lets a model score by recognising the
actor — worth 15–25 points on CREMA-D, and the reason most SER papers
read high.

| Model | Params | Accuracy | Macro-F1 | ms/clip |
|---|---|---|---|---|
| MFCC CNN (the branch's own model, retrained) | 391 K | 0.3147 | 0.3218 | 0.6 |
| **WavLM-base+** | 95.0 M | **0.4901** | **0.4988** | 16.5 |

The baseline is imported from `../audio_cnn.py` unchanged, so the
+17.7 macro-F1 gap isolates the model from the data.

As with the face modality, the averaged number hides the finding.
Split by domain, WavLM scores **0.6938 on studio audio** and **0.3563
on MELD** — and `disgust` and `fear` essentially do not work in the
wild at all. The fusion engine is handed the *wild* per-class figures,
because a laptop microphone in a room is the MELD condition, not the
recording booth's.

### Does three beat two?

MELD ships a transcript per clip, so its test split is the one place
here where two real modalities describe the same utterance. Scored on
those 2,523 clips (face simulated from its measured confusion matrix):

| Modalities | Accuracy | Macro-F1 | ECE |
|---|---|---|---|
| face | 0.5220 | 0.4209 | 0.183 |
| face + audio | 0.5363 | 0.4329 | 0.159 |
| face + text | 0.5355 | 0.4276 | 0.183 |
| **face + audio + text** | **0.5545** | **0.4431** | 0.156 |

Each modality added helps, and audio also *improves calibration* — it
is the best-calibrated leg (ECE 0.035 alone). The trimodal gain over
the best single modality is **+2.2 macro-F1**, which is modest, and
modest for an honest reason: on real pairs the modalities fail
together. A quiet delivery and a flat sentence co-occur. Synthetic
pairing, which `eval_fusion.py` has to use, cannot see that and reports
a larger gain.

Composite score: **49.7 / 100**, broken into capability, fusion gain,
calibration, robustness and conflict discrimination in
[`results/multimodal.json`](audio-emotion-module/results/multimodal.json).

---

## Fusion

A **weighted logarithmic opinion pool** — a weighted geometric mean of
the active probability vectors:

```
log s[c] = Σ_m  e_m · r_m[c] · (log p_m[c] − mean_c log p_m[c])
```

An arithmetic average can never be more confident than its inputs. The
geometric pool multiplies, so a class both modalities find plausible is
reinforced and one either confidently rules out is suppressed — the
right form when the modalities are close to conditionally independent
given the true emotion.

The exponent `e_m` is how much this particular observation is worth:

```
e_face  = 0.8 · quality                      (detector confidence, crop size,
                                              blur, exposure, head pose,
                                              temporal stability)
e_audio = w · recency · voiced                (0.5^(age/8s); voiced fraction
                                              of the window, 0 below 10%)
e_text  = 1.0 · recency · informativeness    (0.5^(age/30s); min(1, tokens/8))
```

The base weights are tuned against a mix of mismatch rates. Face sits
below text because it measures materially weaker cross-dataset.

**The key asymmetry is recency.** The camera reports the face
continuously, so face evidence is always about *now*. A typed message is
a single observation at a single instant, and its relevance decays.
Without that term one message would pin the fused label for the rest of
the session. The UI shows the decay happening.

Audio sits between the two. A voice window is a measurement of how
someone sounded during those four seconds, so it decays far faster than
text — an 8-second half-life against text's 30. Its quality term is the
**voiced fraction** of the window, which plays the role face quality
plays: the encoder returns a confident distribution for a window of
silence just as readily as for a shouted sentence, and only this says
which one it was. Below 10% voiced the window is dropped outright
rather than scaled, so a run of pauses cannot accumulate into a
confident wrong answer.

Per-class reliability `r_m[c]` weights each modality by how well it
actually performs on that class, so each leads where it is strong.

### Does it work?

No paired face+text corpus exists — nobody recorded a person's face and
their message at the same instant with one agreed label. So the
evaluation is a **controlled simulation**: real per-modality confusion
profiles, synthetic pairing, with the mismatch rate as the independent
variable. Crucially nothing is *trained* on the synthetic pairs.

*(The earlier `Training_program_cycling.py` paired an arbitrary face
with an arbitrary sentence by `i % len(files)` and then trained on the
result — which teaches the fusion layer the very assumption it was
supposed to test. That script is superseded and not used.)*

**Fusion only helps if the modalities are calibrated:**

| confidence gap | best single | fused | gain |
|---|---|---|---|
| 0.0 | 65.72% | 57.30% | **−8.43** |
| 1.0 | 67.58% | 66.88% | −0.70 |
| 1.5 | 68.83% | **70.67%** | **+1.85** |
| 3.0 | 71.23% | 74.25% | +3.02 |

"Confidence gap" is how much less peaked a wrong prediction is than a
right one. At zero — a model equally confident whether right or wrong —
no weighting scheme can beat the better single modality, and the table
shows fusion *losing* 8.4 points. **Calibration is not a finishing
touch here; it is the precondition.** Both modalities are temperature
scaled, which is what moves the system out of the top row.

**And the gain holds as the modalities drift apart:**

| mismatch | best single | fused | gain |
|---|---|---|---|
| 0% | 68.83% | 70.67% | +1.85 |
| 10% | 61.88% | 64.90% | +3.02 |
| 20% | 56.45% | 59.85% | +3.40 |
| 50% | 39.92% | 44.57% | +4.65 |

The gain is real but modest — **+1.9 to +6.8 points**. The face
modality is much the weaker of the two (33% macro-F1 cross-dataset
against text's 60%), and fusing a weak signal with a strong one cannot
produce a large jump. What it does produce is complementarity: the face
is wrong about different things than the text is, and per-class
reliability weighting lets each lead where it is stronger. As the text
drifts away from what the face shows, the gain over text-only *grows*,
because the face becomes the only signal still describing the present.

That is also why the conflict flag and the recency decay matter — they
are what stop a stale or contradictory message being treated as current
evidence.

---

## Interface

Served at `http://localhost:8000/` — one page, no build step.

| Panel | What it tells you |
|---|---|
| **Fused Emotion** | The combined answer, its confidence, and which modalities produced it |
| **Modality Contribution** | Face/text split bar plus every factor that set those weights |
| **Text Channel** | Chat log; each message carries its own emotion chip |
| **Face / Text Modality** | Each modality's full seven-class distribution |
| **Conflict banner** | When the modalities name different emotions *and* their distributions genuinely do not overlap |

The contribution breakdown is the point: a weight is only useful if you
can see why it was assigned. Blur the camera and the face weight drops
with `sharpness` named as the culprit; leave a message alone for a
minute and watch its influence decay.

---

## Tests

```bash
cd vit-emotion-v2-deployment && python test_deployment.py    # 38 tests
cd text-emotion-module      && python test_text_module.py    # 33 tests
```

Covering alignment geometry (roll removal, scale invariance), quality
scoring, temporal smoothing, the fusion pool's invariants and
degradation paths, the Ekman mapping, and calibration.

Reproduce the tables:

```bash
cd vit-emotion-v2-deployment
python prepare_eval_data.py              # builds the controlled eval set
python eval_face.py --data eval_data/webcam
python eval_fusion.py
```

---

## Limitations

* **The face model is the bottleneck.** Inference-time fixes recover
  part of the real-world gap; closing it properly means retraining on
  better labels.
* **The face+text fusion evaluation uses synthetic pairing.** It
  characterises the fusion rule; it is not a multimodal benchmark, and
  none is possible with the data this project has for that pair. The
  audio+text pair is different: MELD ships a transcript per clip, so
  `audio-emotion-module/eval_multimodal.py` scores those two on real
  pairs. The face leg of that evaluation is still simulated from the
  deployed model's measured confusion matrix, because no corpus here
  carries a face aligned to a MELD utterance.
* **Text evidence is one message**, not a conversation. The newest
  replaces the previous, because the question is "what is the user
  feeling now".
* **One face at a time** — the largest in frame.
* **The audio modality is trained but not yet live.** `fusion.py`
  accepts audio evidence and the pool weights it, but `server.py` and
  the web UI do not capture microphone audio — so a running session is
  still face + text. Wiring the browser's audio stream to a rolling
  4-second window is the remaining integration step.
* Emotion recognition from appearance estimates *expression*, not what
  a person is actually feeling. Confidence scores should not be read as
  certainty about someone's internal state.

---

## License

See [LICENSE](LICENSE).
