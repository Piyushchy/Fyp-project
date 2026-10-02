# Retraining the face model

```bash
python prepare_face_data.py --out data_v4            # merge four datasets
python train_face.py --data data_v4 --epochs 4 \
    --workers 0 --out vit-emotion-v4.onnx            # ~20 min/epoch on an RTX 4060
```

`--workers 0` is not a performance choice. On Windows each DataLoader
worker is a fresh process that re-imports torch, and the default of 4
died with `OSError [WinError 1455]: the paging file is too small` on
this machine before it reached step 1.

---

## v4: why there was a second retrain

v3 answered **`angry`, 92% confident, on a visibly neutral face** in
live use. The deployment side of that had three contributing bugs (a
crop far tighter than training framing, a temperature below 1.0 that
sharpened rather than softened, and no prior correction) and fixing
those took neutral F1 from 0.000 to ~0.43 on unseen faces. But they
were compensation. The cause was in the weights, and the weights came
from here.

### The labels were wrong

Hold out a model that never saw a given source, and ask what it votes
for on that source's folders. `happy` is the control: it holds at
71-88% everywhere, so a source that collapses on one class is not
merely out of domain.

| folder | ferplus | fer2013 | affectnet | rafdb |
|---|---|---|---|---|
| **angry** | 78.4% | 68.1% | 31.0% | **15.4%** |
| disgust | 69.6% | 92.7% | 17.7% | **12.0%** |
| neutral | 44.1% | 70.3% | 54.3% | 52.4% |
| happy | 86.7% | 87.7% | 71.1% | 75.6% |

RAF-DB's `angry` folder draws **more `neutral` votes than `angry` ones**
— 38% of it reads as neutral. v3 was trained on 4,071 of those images.
It did not develop an angry bias; it was taught one.

The judge is in-domain on FER2013, so that column is flattered. What
survives that caveat is the *shape* of the disagreement, which is why
`rafdb/angry` and `rafdb/disgust` are dropped in `EXCLUDE_PAIRS` while
`affectnet/disgust` — low-scoring but not collapsing into one specific
wrong class, and judged by a model weak at disgust — is kept.

### The fixes

1. **FERPlus added** (`deanngkl/ferplus-7cls`, 35,481 images). FER2013's
   images re-annotated by 10 crowd workers and resolved by majority
   vote. It is the cleanest source measured, and it contributes 12,992
   neutral images — the class that was failing.
2. **RAF-DB angry and disgust dropped** as mostly label noise.
3. **Randomised framing in training** (`random_framing`). Every source
   stores a whole dataset image; deployment hands the model a detector
   box. v3 scored neutral F1 0.56 on images passed uncropped and 0.000
   on the deployed crop of *the same images* — the expression had not
   changed, the framing had. Training now samples 55-100% sub-crops,
   off centre, so the deployed framing is inside the training
   distribution rather than outside it.
4. **Class weights capped at 4.0.** Dropping noisy pairs left `disgust`
   small enough that uncapped inverse frequency would hand it a weight
   near 11, which buys recall by firing on everything — the same shape
   of failure being removed.

## Three attempts to beat v4, and what their failure established

v4 is still the default. Three separate interventions were tried
against it and all three measured worse on the untouched test split.
The interventions were not arbitrary - each targeted a hypothesis the
evidence supported at the time - and the pattern in how they failed is
more informative than any of them individually.

| model | change | accuracy | macro-F1 | neutral F1 |
|---|---|---|---|---|
| **v4** | baseline | **52.29%** | **49.70%** | **44.44%** |
| v5 | balanced sampling | 48.86% | 44.90% | 37.11% |
| v6 | label cleaning | 48.00% | 44.60% | 30.11% |
| v3+v4 | ensemble | 51.45% | 49.74% | 44.68% |

**The pattern: everything that improved the in-domain fit cost
cross-domain accuracy.** v6 is the clearest case - dropping 4,614
confidently-mislabelled images raised validation accuracy from 77.53%
to **81.13%** and left FER2013 unchanged at ~68.5%, while unseen-face
accuracy fell 52.29% -> 48.00%. The cleaning rule required both
`P(stated label) < 0.10` and `P(some other class) >= 0.55`, specifically
so that merely *hard* images would survive. It was not strict enough:
what it removed was disproportionately the ambiguous examples, and
those are what teach a model to generalise.

The ensemble result points the same way. Three checkpoints trained on
different corpora, and averaging them did not beat the best single one
- so their errors are correlated. Independent weaknesses average out;
shared ones do not.

Four observations now converge:

* ensembling differently-trained models does not help
* removing label noise helps in-domain and hurts out-of-domain
* class rebalancing hurts both
* disgust holds 50% F1 in-domain and collapses to ~20% cross-dataset

That is a domain gap, not a capacity, class-balance or annotation
problem. The models have learnt these four datasets about as well as
these four datasets can be learnt. More cleaning, more epochs, bigger
backbones and more checkpoints all attack constraints that are not
binding.

The one lever left is genuinely more diverse faces: the full AffectNet
(~287k against the 27,823 public subset) is license-gated and needs an
authenticated account. Failing that, ~52% is what honest seven-class
cross-dataset FER costs here, and the literature's 40-55% range for the
same setting says that is the right order of magnitude.

## v5: balanced sampling, and why it is not the default

v4's `disgust` scored 19.7% F1, from 54.5% precision and 12.0% recall -
accurate when it fired, and it almost never fired. That is the
signature of a class the model has learnt but is reluctant to predict,
so v5 attacked the reluctance: `WeightedRandomSampler` at alpha=0.5
(disgust 3.1% -> 7.0% of samples per epoch) and the loss-weight cap
raised from 4.0 to 12.0 so it stopped binding.

It made things worse, on both axes:

| disgust | precision | recall | F1 |
|---|---|---|---|
| v4 | 54.5% | 12.0% | 19.7% |
| v5 | 20.0% | 2.0% | **3.6%** |

and overall, on the untouched test split, v5 is 48.86% / 44.90%
macro-F1 against v4's 52.29% / 49.70%.

Precision collapsing alongside recall rules out "still too shy". The
likely cause is memorisation: alpha=0.5 shows each of the 3,346 disgust
images roughly 2.3x per epoch, and over five epochs the model fits
those particular faces rather than the expression.

**This run is confounded and cannot settle it.** The original v5 run
was killed at epoch 3 and resumed with `--init-from` at a lower LR,
which restarted the OneCycle schedule - so v5 is two epochs under one
schedule plus three under another, not five clean epochs. A schedule
disruption would depress every class, and every class did drop
slightly; only disgust collapsed, which points at the sampler. But the
two changes are tangled and the honest statement is that the
experiment does not separate them.

A clean test is one uninterrupted run at alpha=0.5, and if that still
regresses, alpha is not the lever and the 3,346 images are simply too
few for this class.

### Results, on a split nothing was fitted on

`make_clean_splits.py` partitions the unseen evaluation set into
`calib/` and `test/`. Calibration fits a temperature and seven bias
terms on `calib` only; `test` is scored once, at the end, and nothing —
not the crop geometry, not the roll threshold, not the bias — was
chosen using it.

| model | accuracy | macro-F1 | neutral F1 |
|---|---|---|---|
| v2 | 32.86% | 32.48% | 19.35% |
| v3 | 48.00% | 46.86% | 38.55% |
| **v4** | **52.29%** | **49.70%** | **44.44%** |

Earlier versions of this table reported numbers measured on rows the
calibration had been fitted on, which is why v3 and v4 looked closer
than they are.

### The FER2013 test split was never fully held out

FER2013 contains duplicated images, and this project draws its train
and test halves from two different uploads of it. Hashing one against
the other finds **568 of the 7,178 test images already in training** —
7.9% overall, and 27.9% of both `surprise` and `disgust`. Every
FER2013 test number this project has ever quoted, for every model, is
inflated by that.

`make_clean_splits.py` writes `data_v4/fer2013_test_clean` (6,610
images) with the overlap removed. v4 scores **66.61% / 62.58% macro-F1**
on it with a box crop, against the 68.24% it reported on the
contaminated version.

### The two test sets disagree about the crop, and that is the point

On de-duplicated FER2013, the box crop beats the aligned warp:

| config | acc | macro-F1 |
|---|---|---|
| box + 25% | **66.61%** | **62.58%** |
| aligned | 62.65% | 56.99% |

On faces composited into 640×480 frames with background, it reverses:

| config | acc | macro-F1 | neutral F1 |
|---|---|---|---|
| box + 25% | 44.00% | 38.69% | 21.58% |
| aligned | 44.00% | **40.55%** | **32.77%** |

FER2013 images are *already* tight face crops — no background, no neck,
face filling the frame. Detecting and re-framing one moves it away from
its native framing, so the crop that disturbs it least wins almost by
construction. A webcam hands the model a head in a room, not a
pre-cropped portrait.

So preprocessing decisions belong to whichever set shares deployment's
**geometry**, not its content. The deployment README's advice used to be
the opposite of this and it is what produced the original crop
settings.

### What more data would and would not buy

AffectNet was already in at full resolution — sampled source images run
375–627 px against the 224 written here, so it was never the
bottleneck. Three higher-resolution pulls were investigated and none
helps: `Piro17/affectnethq` is license-gated (anonymous download 401s),
`Piro17/dataset-affecthqnet-fer2013` is downscaled to 48×48, and
`Mauregato/affectnet_short` is 96×96. Resolution was not the
constraint. Label quality was, and FERPlus is what addresses it.

Corpus: **107,536 images** (v3: 77,003). Neutral 15,229 → 28,221.

| class | v3 | v4 |
|---|---|---|
| angry | 11,704 | 10,757 |
| disgust | 3,973 | 3,346 |
| fear | 8,074 | 8,903 |
| happy | 18,217 | 27,582 |
| neutral | 15,229 | 28,221 |
| sad | 10,720 | 15,147 |
| surprise | 9,086 | 13,580 |

`train_face_vit.ipynb` is the Colab/Kaggle version for machines without
a local GPU.

---

## Why retrain

The deployed model is not broken. Run it on FER2013's test split with a
plain resize and it scores **74.40% accuracy / 73.72% macro-F1** —
better than the 67.69% the project used to claim.

The problem is everything that is not FER2013:

| dataset | accuracy |
|---|---|
| FER2013 test — the model's own distribution | **74.40%** |
| RAF-DB — never seen in training | **51.80%** |

A 23-point drop on unseen faces. That is what a webcam user
experiences, and no amount of preprocessing tuning touches it. The
model learned FER2013, not facial expression.

Before concluding that, it is worth ruling out the cheaper
explanations, which the deployment-side work did:

* **Preprocessing?** No — the deployed mean/std 0.5 is the best of
  three candidates tried (ImageNet normalisation scores 52%).
* **Crop geometry?** No — a plain box crop already beats every aligned
  variant on upright faces. See the deployment README.
* **Calibration?** It fixes the confidence, not the accuracy.

That leaves the training data.

---

## The corpus

Three datasets that all annotate the same seven emotions, merged onto
the deployment's label space:

| source | images | what it contributes |
|---|---|---|
| FER2013 train | 28,709 | volume, and the distribution the current model already knows |
| AffectNet | 27,823 | real photographs, in-the-wild lighting and pose |
| RAF-DB | 20,471 | real photographs, crowd-annotated |
| **total** | **77,003** | |

FER2013's **test** split (7,178 images) is written to a separate
directory and never trained on, so the headline number is directly
comparable to the deployed model's 74.40% on those exact images.

### Class balance

```
angry  11,704   disgust  3,973   fear  8,074   happy 18,217
neutral 15,229  sad     10,720   surprise 9,086
```

`disgust` is 5% of the corpus and `happy` is 24%, so the loss is
inverse-frequency weighted. Without that, a model can ignore `disgust`
entirely and still look respectable on accuracy — which is precisely
why macro-F1 is the metric that selects the epoch.

### Label-order guards

Each source's class order is asserted against the downloaded features
before anything is written. A silent reordering upstream would not
crash anything; it would just mislabel a third of the corpus and train
a perfectly healthy classifier on nonsense.

---

## What the training does

* **Class-balanced cross-entropy with label smoothing.** The smoothing
  stands in for FER+ style soft labels: these datasets ship one hard
  label per image, but the underlying human judgements are genuinely
  uncertain, and training as if they were not makes the model
  overconfident in exactly the way calibration then has to undo.
* **Webcam-domain augmentation** — horizontal flip, small rotation and
  scale, brightness and contrast jitter, Gaussian blur, JPEG
  compression artefacts, sensor noise, random occlusion, and occasional
  grayscale. The last is there deliberately: it stops the model keying
  on skin tone, which is both a fairness concern and a generalisation
  one.
* **Mixed precision (bf16) on CUDA**, which is what makes 77k images
  over three epochs a half-hour job rather than an overnight one.
* **Temperature fitted on validation**, never on the held-out test
  split, and written into the exported config.

The architecture, the seven-class head and the ONNX signature are all
unchanged from v2. That makes the comparison clean: any difference is
attributable to data, not capacity.

---

## Installing the result

**Already done for the checkpoint in this repository** — v3 is the
deployed default. These are the steps if you retrain:

1. Copy `vit-emotion-v3.onnx` **and** `vit-emotion-v3.onnx.data` into
   `../vit-emotion-v2-deployment/`. ViT-base exceeds the 2 GB protobuf
   limit, so the weights live in the sidecar file and the two must
   travel together.
2. Copy the exported `emotion_config.json` over the deployment's — it
   carries the fitted `temperature`. Keep the old one as
   `emotion_config.v2.json`; `EMOTION_MODEL=v2` reads it.
3. Put the per-class F1 **measured on unseen faces** into
   `reliability.json` under `face.per_class_f1`. Not the FER2013
   figures: fusion weights each class by how much it can be trusted in
   deployment, and FER2013 overstates that by 20 points.
4. Re-run the checks:

```bash
cd ../vit-emotion-v2-deployment
python test_deployment.py                       # label-order guard included
python eval_face.py --data <your test dir>
python eval_fusion.py --refresh-reliability
```

`test_shared_space_matches_the_exported_model` will catch a label-order
mistake immediately, which is the failure mode most likely to slip
through unnoticed.

---

## Result

3 epochs, 77k images, ~30 min on an RTX 4060.

| test set | v2 (FER2013 only) | v3 (3 datasets) | delta |
|---|---|---|---|
| FER2013 test — **v2's home turf** | 72.90% / 72.77% | 69.04% / 66.20% | −3.9 / −6.6 |
| **A 4th dataset, unseen by both** | 40.57% / 36.74% | **48.00% / 46.61%** | **+7.4 / +9.9** |

*(accuracy / macro-F1)*

**v3 is worse on FER2013 and better on everything else.** That is the
trade the retrain was for, and it is worth stating in both directions
rather than quoting only the favourable half.

v2 was trained on FER2013, so FER2013 flatters it; a model that spends
part of its capacity on AffectNet and RAF-DB necessarily gives some of
that back. What it buys is the number in the second row, and the second
row is the one that describes a stranger sitting at a webcam.

The per-class picture on unseen faces is where the difference is
starkest — v2 had essentially given up on several classes:

| | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| v2 | 41.4 | **5.5** | 40.1 | 79.1 | 23.4 | 38.7 | 29.0 |
| v3 | 43.6 | **28.8** | 37.3 | 80.0 | 36.9 | 56.7 | 43.0 |

`disgust` goes from 5.5 to 28.8 F1, `sad` from 38.7 to 56.7. Still not
good — but the difference between "unreliable" and "never correct"
matters for a system that has to pick one of seven answers.

v3 is the deployed default. Set `EMOTION_MODEL=v2` to switch back and
compare.

### Why the absolute numbers are low

48% on seven classes is well above the 14% chance rate and well below
anything that should be described as solving the problem. Facial
expression recognition across datasets is genuinely hard: annotators
disagree, the label taxonomies do not mean quite the same thing in each
corpus, and "the expression a stranger is making" is not a
well-posed question with one right answer. Treat the output as an
estimate of expression, not a readout of feeling.

---

## Honest limits

* **AffectNet and RAF-DB here are public mirrors**, not the official
  releases, which are gated behind access requests. Check their terms
  before using this in anything published.
* **Three datasets is not the same as three independent samples of
  humanity.** All three over-represent certain demographics, and
  merging them does not fix that — it only broadens the imaging
  conditions.
* **No pose or occlusion labels**, so nothing here measures whether the
  model fails more on profile views or on glasses, only that it fails
  less overall.
* Emotion recognition from appearance estimates *expression*, not what
  a person is feeling. A better score on any of these datasets means
  better agreement with annotators, not better mind-reading.
* **The unseen test set is only 700 images.** It is the only genuinely
  held-out set available once three datasets are spent on training, and
  700 images over seven classes means each per-class figure rests on
  roughly a hundred examples. The direction of the result is clear; the
  precise magnitudes are not.
* **Three epochs.** Validation macro-F1 was still climbing (68.1 -> 70.8
  -> 71.9), so this is under-trained rather than converged. More epochs
  is the cheapest remaining improvement.
