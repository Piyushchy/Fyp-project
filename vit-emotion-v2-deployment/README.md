# Multimodal Emotion Recognition — Deployment

Real-time emotion recognition from **two modalities at once**: a Vision
Transformer reads the face from a webcam, a fine-tuned compact
transformer reads typed messages, and a reliability-weighted pool fuses
them into one answer that explains how it was reached.

```
webcam ──▶ YuNet detect ──▶ adaptive crop ──▶ ViT ONNX ──▶ smooth ──┐
                                                                     ├──▶ fused emotion
chat  ──▶ TinyBERT (GoEmotions, 7-class) ──▶ decay with age ─────────┘
```

Both modalities predict over the **same seven classes**, so their
probability vectors can be compared directly:

```
angry · disgust · fear · happy · neutral · sad · surprise
```

---

## Quick start

```bash
pip install -r requirements.txt
python setup_models.py                 # one-time: YuNet detector + ViT weights
python server.py --device gpu          # omit --device to allow a CPU fallback
```

Then open <http://localhost:8000>.

`setup_models.py` downloads two things: the YuNet detector (227 KB) and
`vit-emotion-v4.onnx.data` (328 MB). The weight blob is not in the
repository because GitHub rejects files over 100 MB — it is attached to
the [`face-model-v4` release][release] instead, and the small `.onnx`
graph file beside it *is* committed, so a clone is missing only the
weights. Add `--all-weights` to also fetch v6.

[release]: https://github.com/Piyushchy/Fyp-project/releases/tag/face-model-v4

Which checkpoint is served:

| `EMOTION_MODEL` | accuracy on unseen faces | notes |
|---|---|---|
| **`v4`** (default) | **52.29%** | best measured; what `setup_models.py` fetches |
| `v6` | 48.00% | label-cleaning experiment — better in-domain, worse on new faces |
| `v2` | 32.86% | the original FER2013-only model, committed in full |

Those numbers are measured on 350 faces from a dataset none of the
models trained on, with calibration fitted on a disjoint split. See
[`../training/README.md`](../training/README.md) for why v6 scores higher
on validation yet ships as an experiment rather than the default.

### GPU

Both modalities use the GPU when one is available, and fall back to CPU
when it is not. `GET /health` reports which, and the server now logs a
**warning** on fallback rather than an info line.

```bash
pip install onnxruntime-gpu                                  # face
pip install torch --index-url https://download.pytorch.org/whl/cu130   # text
```

Measured on an RTX 4060, single 224² frame: **303 ms CPU → 6.6 ms
CUDA**. End to end through the WebSocket that is **0.8 → 37 FPS**.

#### Choosing the device

```bash
python server.py                  # auto: GPU if available, else CPU
python server.py --device gpu     # refuse to start if no GPU provider
python server.py --device cpu     # force CPU
python server.py --model v3       # compare checkpoints
```

`--device gpu` exists because the failure this project actually had was
a *silent* fallback: everything worked, nothing errored, and the only
symptom was 0.8 FPS. Asking for the GPU explicitly means being told
when you did not get it. `EMOTION_DEVICE` / `EMOTION_MODEL` do the same
thing as the flags.

The header now carries a **device chip** showing the active provider,
amber when the face model is on CPU. `GET /health` reports
`model`, `device_requested`, `face_provider`, `text_device`, `gpu`,
`crop_mode` and `multicrop`.

The standalone `realtime_inference_onnx.py` takes the same `--device`
and `--model` flags. It used to carry its own copy of the inference
path — v2 weights with v3's config, Haar detector, a 5% crop, no
calibration, CPU hardcoded — which reproduced every bug the deployment
had already fixed. It now calls the same `FacePipeline` the server
does.

#### Install exactly one onnxruntime wheel

`onnxruntime` and `onnxruntime-gpu` unpack into the *same* `onnxruntime`
package directory. Installing both leaves whichever pip touched last in
place, and it overwrites the other's `onnxruntime_pybind11_state.pyd`.
Nothing errors: the CUDA provider `.dll` is still on disk, so the only
symptom is that `get_available_providers()` quietly stops listing
`CUDAExecutionProvider` and every frame runs on CPU.

Check and repair:

```bash
pip list | grep onnxruntime      # must print ONE line
python -c "import onnxruntime; print(onnxruntime.get_device())"   # want GPU
pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu
```

The other wrinkle: ONNX Runtime's CUDA provider needs cuDNN and cuBLAS
on the DLL search path, and on Windows it will fall back to CPU without
saying why if they are missing. A CUDA-enabled torch already ships
exactly those DLLs, so `face_pipeline._register_cuda_dlls()` points the
loader at `torch/lib` rather than making you install the CUDA toolkit
separately.

Set `EMOTION_DEVICE=cpu` to force CPU — worth doing when comparing
against the CPU latency figures quoted here.

Then open **http://localhost:8000/**.

Serve the page from the server rather than opening `index.html` from
disk — the client derives the WebSocket address from `window.location`,
and a `file://` URL has no host to derive it from.

The text modality needs a trained checkpoint:

```bash
cd ../text-emotion-module
python train.py --task goemotions --model tinybert     # ~26 min on CPU
```

Without one the server still starts and runs face-only; the UI says so
rather than showing an inert text box.

---

## What the interface shows

| Panel | What it tells you |
|---|---|
| **Fused Emotion** | The combined answer, its confidence, and which modalities produced it |
| **Modality Contribution** | A face/text split bar plus every factor that set those weights |
| **Text Channel** | Chat log; each message carries its own emotion chip and confidence |
| **Face / Text Modality** | Each modality's full seven-class distribution, independently |
| **Conflict banner** | Appears when the two modalities name different emotions and their distributions genuinely do not overlap |

The face crop defaults to `CropMode.ADAPTIVE` — plain box crop on an
upright face, eye-line aligned warp past ~18° of roll. See
[Measured results](#face-what-the-crop-geometry-is-actually-worth) for
why neither fixed policy wins on its own.

---

## How the fusion works

A **weighted logarithmic opinion pool** — a weighted geometric mean of
the two probability vectors:

```
log s[c] = Σ_m  e_m · r_m[c] · (log p_m[c] − mean_c log p_m[c])
p_fused  = softmax(log s / T)
```

An arithmetic average can never be more confident than its inputs and
keeps mass on any class either modality merely tolerates. The geometric
pool multiplies: a class both modalities find plausible is reinforced,
one that either confidently rules out is suppressed. That is the right
form when the modalities are close to conditionally independent given
the true emotion — which face pixels and typed words largely are.

### The exponent `e_m` — how much this observation is worth

```
e_face = 0.8 · quality
e_text = 1.0 · recency · informativeness
```

The base weights are tuned by `eval_fusion.py --tune` against a mix of
mismatch rates. Face sits below text because it measures materially
weaker cross-dataset — it is worth 0.8 of a calibrated observation, not
a full one.

| Factor | Question it answers |
|---|---|
| `quality` | Is this frame's face worth looking at? Detector confidence, crop size, blur, exposure, head pose, temporal stability — combined as a weighted geometric mean, so one bad factor is not outvoted by three good ones |
| `recency` | How long ago was this typed? `0.5 ^ (age / 30s)`, hard zero past 180s |
| `informativeness` | Is there enough text to carry an emotion? `min(1, tokens / 8)` |

**Note what is deliberately absent: the prediction's own confidence.** A
log pool already handles that for free — a uniform `p_m` has a flat log
vector, which is a constant offset and cancels exactly in the softmax.
A modality that is merely guessing therefore contributes nothing
without being told to. Adding a certainty term to the exponent would
discount the same uncertainty twice and, at `e_m < 1`, would flatten a
single trustworthy modality below its own honest confidence.

Certainty *is* computed — it scales the **influence** bar in the UI, so
a present-but-uninformative modality reads as ~0% — but it steers
attribution, not the arithmetic.

### Why the log-probabilities are centred

Per-class reliability `r_m[c]` scales each modality by how well it
actually performs on that class. Because `r` varies across `c`, an
uncentred `r[c]·log p[c]` would stop a uniform `p` from cancelling —
`log(1/7)` is negative, so high-reliability classes would be pushed
*down* by a modality that said nothing at all. Subtracting the mean
log-probability first makes a uniform input contribute exactly zero for
any `r`, and changes nothing when `r` is flat.

### Conflict

`agreement` is the Bhattacharyya coefficient. Its practical range is
much narrower than [0,1] — over seven classes the shared tails floor it
near 0.4 even for fully opposed peaked distributions:

| face / text | coefficient |
|---|---|
| happy .8 / happy .8 | 1.00 |
| happy .5 / neutral .5 | 0.83 |
| happy .6 / angry .6 | 0.73 |
| happy .7 / angry .85 | 0.52 |
| happy .99 / angry .99 | 0.09 |

So the threshold sits at **0.80**, paired with an argmax-differs test.
Either condition alone misfires: a low coefficient with the same argmax
just means the modalities are differently confident about the same
emotion, and differing argmaxes with a high coefficient means both are
near-uniform — two shrugs, not a conflict.

All of this is data, not code: see [`reliability.json`](reliability.json).

---

## Measured results

### Face: what the crop geometry is actually worth

> **This section was wrong in an earlier revision and has been
> rewritten.** The original claimed eye-line alignment was the largest
> available accuracy lever, worth about +4 macro-F1. That conclusion
> came from comparing two configurations on composited,
> out-of-distribution images where *both* scored badly — the difference
> between them meant nothing. Measured properly, alignment is a net
> loss on upright faces. The corrected numbers are below.

#### First: is the model or the pipeline at fault?

`python eval_face.py` on FER2013's test split, no detector, plain resize:

| preprocessing | accuracy | macro-F1 |
|---|---|---|
| **config mean/std 0.5 (what we deploy)** | **74.50%** | **72.83%** |
| ImageNet mean/std | 52.25% | 48.48% |
| no normalisation | 69.08% | 65.01% |

The deployed preprocessing is correct, and the model is *better* than
the 67.7% the project README claimed. So neither the weights nor the
config were the problem.

#### Then: does alignment help?

Same weights, same images, varying only the crop:

| configuration | FER2013 acc | FER2013 macro-F1 |
|---|---|---|
| raw resize, no detector | **71.93%** | **70.63%** |
| detector + box crop (legacy) | 70.76% | 69.19% |
| detector + aligned warp, best of 40 swept geometries | 66.55% | 63.85% |

Alignment loses at **every cell** of the geometry grid. The warp
resamples an already small crop and replicates edge pixels where the
rotated frame leaves the source; on an upright face there is no pose
error to win back, so that cost is all there is.

The earlier guess of IOD 0.42 / eye-Y 0.40 was itself about 3 points
worse than the sweep's best cell (0.38 / 0.34) — reasoning about
framing conventions is not a substitute for measuring.

**That sweep was invalid, and so was the 0.38 it produced.** It ran on
`eval_data`, which `prepare_eval_data.py` built from RAF-DB's *train*
split — the same rows `training/prepare_face_data.py` feeds v3. Measured
overlap against `training/data/train` is 117/120 neutral, 120/120 angry,
120/120 happy. It also scored at T=1 with no logit bias, a configuration
the server does not serve.

Re-swept on `eval_data_unseen` under the deployed calibration:

| IOD | 0.18 | 0.21 | **0.24** | 0.27 | 0.30 | 0.34 | 0.38 | 0.42 |
|---|---|---|---|---|---|---|---|---|
| acc | 41.4% | 43.9% | **47.1%** | 47.0% | 45.9% | 46.9% | 46.1% | 47.1% |
| macro-F1 | 41.0% | 43.8% | **46.5%** | 46.5% | 45.0% | 46.0% | 45.4% | 46.1% |
| neutral F1 | 39.3% | 36.7% | **37.4%** | 31.6% | 25.4% | 24.1% | 27.7% | 25.2% |

`EYE_DISTANCE_RATIO` is **0.24** and `LEGACY_MARGIN` is **0.25** (was
0.05). The old values framed the face far tighter than either model was
trained on, and `neutral` paid for it: feeding the whole uncropped
image scored neutral F1 0.56 where the deployed crop scored 0.30.

#### But it inverts under roll

Same images with a known roll applied — aligned minus box crop, in
accuracy points. Re-measured on `eval_data_unseen/webcam` under the
deployed calibration:

| roll | 0° | 5° | 10° | 15° | 20° | 25° | 30° |
|---|---|---|---|---|---|---|---|
| **delta** | −0.87 | +0.58 | **+1.90** | **+2.63** | **+2.50** | **+2.97** | **+2.84** |

Crossover just under **10°**, and `ROLL_ALIGN_THRESHOLD_DEGREES` is
10.0. So neither fixed policy is right, and the deployed default is
`CropMode.ADAPTIVE`: measure the roll from the eye line, take the plain
box crop below the threshold and the aligned warp above it.

The earlier version of this table put the crossover at 18° and had the
warp losing 3 points at 10°. It was measured at T=1 with no logit bias.
A logit bias is not a monotone rescaling, so it reorders predictions
and moves the crossover — scoring a configuration nobody serves
answers the wrong question. The same mistake set the crop geometry
(below).

#### Calibration

Temperature scaling cuts expected calibration error and, on its own,
changes no prediction. It matters because fusion pools
log-probabilities, so an overconfident modality wins arguments it has
not earned.

Two corrections to what this section used to claim:

**Temperature must be fitted on the deployment distribution.** v3
shipped with `temperature: 0.7664`, fitted on FER2013 test. Below 1.0 a
temperature *sharpens*, which is backwards. On unseen webcam faces that
produced ECE 0.301 and ~90% confidence on wrong answers — a calm face
reported as `angry 92%`. Refitted where the model actually runs, T is
**1.45**.

**Temperature alone is not enough.** v3 carries a large prior error: it
over-predicts `angry` and almost never emits `neutral` — 2 predictions
in 693 faces, F1 0.000. Temperature cannot fix that, because it cannot
reorder anything. A per-class `logit_bias`, fitted alongside T on one
half of the unseen set and reported on the other, can:

| configuration | acc | macro-F1 | neutral F1 | mean conf |
|---|---|---|---|---|
| T=0.7664, no bias (shipped) | 44.2% | 0.414 | **0.125** | 68% |
| T=1.45 + per-class bias | **46.6%** | **0.457** | **0.409** | 40% |

The fitted bias is `angry −0.97 … neutral +0.49`, which is the prior
error stated numerically. `EmotionClassifier` had supported
`logit_bias` all along; nothing was passing one.

#### Cross-dataset generalisation is the real problem

| dataset | plain resize | note |
|---|---|---|
| FER2013 test | **74.40%** | the model's own distribution |
| RAF-DB | **51.80%** | never seen in training |

A 23-point drop on unseen faces. *That* is what a user sees on their
webcam, and no amount of crop tuning fixes it — it needs retraining on
more than one dataset, which is what [`../training/`](../training/)
now does.

> An earlier revision of this README quoted ~36% cross-dataset. That
> figure came from `prepare_eval_data.py`, which composites faces into
> synthetic frames and adds webcam distortion. The compositing itself
> was costing ~16 points, so the number described the harness more than
> the model. 51.8% on raw RAF-DB is the honest cross-dataset figure.

### Fusion

`python eval_fusion.py`

There is no paired face+text corpus, so this is a **controlled
simulation**: real per-modality confusion profiles, synthetic pairing,
with the mismatch rate as the independent variable. Nothing is trained
on the synthetic pairs — that distinction is what separates this from
the abandoned `Training_program_cycling.py`, which trained on
index-cycled pairs.

Both modalities are profiled from their **real measured** per-class F1
— the face from `results_face_eval.v3.json`, row `6. deployed`
(cross-dataset on faces trained on by neither model, **48.4% macro**),
the text from `benchmark_goemotions.json` (59.5% macro). The face is
still the weaker modality, and the tables reflect that honestly rather
than assuming parity.

> Previously this read `results_face_eval.json` at 33.3% macro, which
> was wrong twice over: that file was written by a harness hardcoded to
> the **v2** weights while loading **v3's** config, and `eval_fusion.py`
> took its *last* row — `+ calibration`, temperature-only — rather than
> the deployed row. Fusion was therefore tuned against a face modality
> seven accuracy points and thirty-five points of neutral F1 weaker
> than the one it actually receives. With the deployed profile,
> face-only rises 39.9% → 52.9% and the fusion gain at 0% mismatch
> rises +1.82 → **+5.77**.

**Fusion only helps if the modalities are calibrated.** This is the
result that decides whether the approach is sound at all:

| confidence gap | face only | text only | fused | gain |
|---|---|---|---|---|
| 0.0 | 33.15% | 65.72% | 57.30% | **−8.43** |
| 1.0 | 36.85% | 67.58% | 66.88% | −0.70 |
| 1.5 | 39.92% | 68.83% | **70.67%** | **+1.85** |
| 3.0 | 43.73% | 71.23% | 74.25% | +3.02 |

"Confidence gap" is how much less peaked a wrong prediction is than a
right one. At zero — a model equally confident whether right or wrong —
no weighting scheme can beat the better single modality, and the table
shows exactly that: fusion *loses* 8.4 points. Calibration is not a
finishing touch here; it is the precondition. Both modalities are
temperature-scaled (face T≈2.92, text T≈1.20), which is what moves the
system out of the top row.

And because the modalities are asymmetric, fusion helps across the
whole mismatch range rather than only when they agree:

| mismatch | face only | text only | fused | gain | conflict flagged |
|---|---|---|---|---|---|
| 0% | 39.92% | 68.83% | 70.67% | +1.85 | 14.9% |
| 10% | 39.92% | 61.88% | 64.90% | +3.02 | 16.1% |
| 20% | 39.92% | 56.45% | 59.85% | +3.40 | 17.3% |
| 50% | 39.92% | 36.90% | 44.57% | +4.65 | 21.8% |

The gain is real but modest — **+1.9 to +6.8 points over the better
single modality**. A weak modality still carries complementary
information: the face is wrong about different things than the text is,
and per-class reliability weighting lets each lead where it is
stronger. As the text drifts away from what the face shows, the gain
over text-only actually *grows*, because the face becomes the only
signal still describing the present.

That is also why the conflict flag and the recency decay matter: they
are what stop a stale or contradictory message from being treated as
current evidence.

> Earlier revisions of this table, computed against placeholder
> reliability figures rather than measured ones, showed a much larger
> +8.5 gain. The numbers above supersede them.

---

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI: one WebSocket carries both modalities |
| `face_pipeline.py` | detect → align → quality → classify → smooth |
| `fusion.py` | the weighted log-opinion pool |
| `labels.py` | the shared seven-class vocabulary |
| `reliability.json` | per-class F1, base weights, decay half-life |
| `index.html` | dashboard (single file, no build step) |
| `setup_models.py` | fetches the YuNet detector |
| `prepare_eval_data.py` | builds a composited evaluation set (see the caveat above) |
| `eval_face.py` / `eval_fusion.py` | the tables above |
| `test_deployment.py` | 38 tests: geometry, quality, smoothing, fusion |
| `realtime_inference_onnx.py` | the original standalone OpenCV demo |

```bash
python test_deployment.py       # or: pytest test_deployment.py -v
```

39 tests: alignment geometry, adaptive roll dispatch, quality scoring,
temporal smoothing, the fusion pool's invariants and degradation
paths.

---

## Protocol

One WebSocket at `/ws/video` carries both modalities.

**Client → server**

```json
{"type": "frame", "data": "<base64 jpeg>"}
{"type": "text",  "text": "a message", "id": 7}
```

A bare Base64 string is still accepted as a frame, so the original
single-modality client keeps working.

**Server → client**

```json
{"type": "frame", "fps": 12.4,
 "face":   {"detected": true, "bbox": {...}, "quality": {...},
            "label": "happy", "confidence": 0.62, "probs": {...}},
 "text":   {"label": "angry", "age_seconds": 12.4, "probs": {...}},
 "fusion": {"label": "happy", "confidence": 0.58, "mode": "fused",
            "agreement": 0.74, "conflicted": false,
            "weights": {"face": 0.81, "text": 0.19},
            "face": {"factors": {...}}, "text": {"factors": {...}}}}
```

Also: `GET /health` reports which components loaded, and
`POST /api/text/analyze` classifies one message statelessly without
touching any session's fusion state.

---

## Detector fallback

`setup_models.py` fetches **YuNet**, which returns a confidence score
*and* five landmarks. The landmarks are the whole point — without an
eye line there is no alignment, and alignment is the largest
accuracy lever available without retraining.

If the asset is missing the pipeline falls back to OpenCV's bundled
Haar cascade. It still runs, but the boxes are loose, unaligned and
confidence-free — the configuration the original 67.7% was measured
under. The UI marks this with an amber `unaligned` chip.

> The `mediapipe` import in the original `server.py` silently never
> worked on this environment: mediapipe 0.10.35 on Python 3.14 ships
> only `mediapipe.tasks`, with no legacy `solutions` API, so
> `hasattr(mp, "solutions")` was always false and every request took
> the Haar path. That is why `mediapipe_confidence` always rendered as
> `—`. The dependency has been dropped.

---

## Known limitations

* **The face model is the bottleneck.** 67.7% in-dataset, far lower
  cross-dataset. Inference-time fixes recover part of the gap; closing
  it properly means retraining on better labels.
* **The pairing in `eval_fusion.py` is synthetic.** It characterises
  the fusion rule; it is not a multimodal benchmark, and no such
  benchmark is possible with the data this project has.
* **Text evidence is a single message**, not a conversation. The newest
  message replaces the previous one rather than accumulating, because
  the question is "what is the user feeling now".
* **One face at a time** — the largest in frame.
* Emotion recognition from appearance is an estimate of *expression*,
  not of what a person is actually feeling.
