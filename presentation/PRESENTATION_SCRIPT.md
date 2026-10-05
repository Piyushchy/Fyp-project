# Presentation script - Sept eval, multimodal emotion recognition

25 slides, about 24 minutes spoken, which leaves room for questions in a 30-minute slot.
The same text is embedded as PowerPoint speaker notes, so it is visible in Presenter View.

**Long slides (2 min each):** 8, 13, 14, 22, 23. Everything else is 20-90 seconds.

**The one thing not to forget:** slides 6 and 7 describe the earlier encoder study (MentalBERT / WavLM / DINOv2). Only WavLM reached the built system. Say so on slide 6 and reconcile it on slide 14 - if a panel member spots it first, it reads as inflation rather than iteration.

---

## Slide 1 - Title

OPEN (20s)

Good morning. We are building an emotion recognition layer for
telemedicine - a system that watches a consultation and tells the
clinician what the patient's face, voice and words suggest they are
feeling, in real time.

I am <name>, with Muskan, Yashraj and Ankita, mentored by Dr. Girishma
Sharma.

One framing note before I start. Since the last review we have gone from
one modality to three, and the honest headline is that the fused system
beats every single modality and is still not stable enough to put in
front of a clinician. I will show you both halves of that, with numbers.

---

## Slide 2 - Introduction

INTRODUCTION (45s)

Emotion recognition is the computational identification of an emotional
state from signals a person emits without being asked - the face, the
voice, the words, and in some of the literature physiological channels
like EEG.

The field moved from rule-based coding of facial action units to deep
learning, and the practical consequence is that a single model can now
be handed raw pixels or raw audio instead of hand-designed features.

What we are building is the healthcare application of that: a layer a
telemedicine platform can add so the emotional state of a patient is
visible alongside their clinical data, live, during the consultation.

Keep one phrase in mind for later - "in real time" is the hard part, and
it is where our remaining problem lives.

---

## Slide 3 - Motivation

MOTIVATION (60s)

Three beats here.

The problem: telemedicine platforms carry clinical data and speech.
They do not carry non-verbal signal. In a physical consultation a doctor
reads posture, hesitation, a flattened voice - over a video call most of
that is either lost or not attended to, because the clinician is looking
at a chart.

Why it matters: emotional state affects both diagnosis and whether the
patient actually follows the treatment. And mental health deterioration
in particular shows up in affect before it shows up in anything a
patient volunteers.

Our goal: a layer that is real-time, privacy-aware and - the word we
care most about - validated. Plenty of published systems quote 90%+
accuracy. Part of what we will show today is why those numbers usually
do not survive contact with an unseen face.

---

## Slide 4 - Objectives

OBJECTIVES FOR THIS CYCLE (45s)

Four, and I will tell you up front which ones we met.

System integration - connect the finished frontend to the Python ML
backend. Done; it is one WebSocket carrying all three modalities, and
I will demo it.

Establish the multimodal baseline - extract and fuse video, audio and
text. Done, and measured end to end.

Prove the multimodal approach beats single-modality. Done, but the
margin is smaller than we expected, and the reason is interesting
rather than embarrassing. Slide on it later.

Finalise the web platform UI. Done - the therapist-facing portal works,
and it shows not just the answer but which modality produced it.

---

## Slide 5 - Methodology - the Transformer figure

METHODOLOGY - THE TRANSFORMER (45s)

This is the architecture figure from "Attention Is All You Need" - the
encoder-decoder transformer. Do not walk the whole diagram; the panel
has seen it. Use it to make one point and move on:

"All three of our encoders are transformers, and the reason is the
block in the middle - multi-head attention. It lets every part of the
input look at every other part in a single step. For a face that means
the brow and the mouth are related directly rather than through layers
of local filters. For speech it means a word can be interpreted against
the tone of the sentence around it.

We use the encoder half only. We are classifying, not generating, so
there is no decoder stack in anything we ship."

That last sentence pre-empts the obvious question about the right-hand
column of the figure.

---

## Slide 6 - Models Used (comparison table)

MODELS USED - THE COMPARISON TABLE (60s)

This table is the encoder study from the earlier cycle: three candidate
configurations across text, audio and video.

  Baseline       BERT-base / MFCC-40 + CNN / ViT-Base
  Combination A  MentalBERT / WavLM Base+ / DINOv2
  Combination B  MiniLM-L6 / log-mel + MobileNetV3-S / EfficientNet-B0

Read it as the three bets available: a conventional baseline, a
"best-representation" option, and a "smallest viable" option.

IMPORTANT - say this out loud rather than letting them find it:

"This table is the study we did before building. The system we are
demoing today is not Combination A exactly. One finding carried over -
WavLM - and I will show you on slide 14 what we actually shipped in the
other two slots and why. The numbers on the next slide belong to this
earlier study, not to the system you are about to see measured."

If you do not say that, a panel member who compares slide 7's fear F1 of
0.96 with our measured audio results on slide 16 will think we are
inflating. It is the single most dangerous inconsistency in the deck.

---

## Slide 7 - Why Combination A Is Superior

WHY COMBINATION A IS SUPERIOR (60s)

Three reasons from the earlier study - take them in order, briefly.

Domain-matched text: MentalBERT is pre-trained on mental-health
language, so it represents clinical phrasing better than general BERT.

Learned speech representation: WavLM instead of hand-crafted MFCC
features. This is the finding that survived and is in the system today -
a learned representation captures prosody that a fixed filterbank
throws away.

Strong visual representation: DINOv2, a self-supervised visual encoder.
Note the honest caveat already written on the slide - its individual
contribution was never isolated.

THEN THE BRIDGE, and do not skip it:

"Two of these three did not make it into the build. MentalBERT and
DINOv2 are not in our codebase. Neither predicts our seven emotion
classes directly, and both exceed the CPU budget we set for a
deployment that has to run on one machine. What we built instead, and
the measurements behind it, are on slide 14.

So treat these F1 numbers as the earlier comparison's. Every figure
after slide 12 is measured on our own splits and is reproducible from
the repository."

PREPARED ANSWER if asked "why is fear 0.96 here and much lower later?"
Different data, different split policy, different model. Our later
numbers use speaker-disjoint splits, which remove the actor-identity
shortcut. That alone is worth 15 to 25 points.

---

## Slide 8 - Methodology in three steps

METHODOLOGY IN THREE STEPS (2 min - this is a core slide)

Step one, read each signal. A Vision Transformer reads the face from
the webcam. A WavLM speech encoder reads tone of voice. A compact BERT
reads typed text. The important design decision is that each one is
trained and evaluated completely independently, so a weak modality is
visible before it ever reaches the fusion. Nothing is hidden by the
averaging.

Step two, speak one language. All three emit a probability over the
SAME seven emotions, in the same order: angry, disgust, fear, happy,
neutral, sad, surprise. That shared label space is what makes fusion
arithmetic rather than translation - position four means happy in all
three vectors. Getting the audio model to speak those seven directly
took rebuilding its training corpus, which is slide 15.

Step three, weigh and decide. A weighted logarithmic opinion pool - a
geometric mean, not an average. Geometric because the modalities are
roughly independent given the true emotion, so a class both find
plausible should be reinforced and a class either rules out should be
suppressed. An arithmetic average can never be more confident than its
inputs; a product of experts can.

And the weights are not fixed. They track how much THIS observation
deserves trust: face crop quality, how recently the voice was heard,
how long the typed message was.

The numbers on the right: 52% face, 50% voice, 33% text measured alone
on data they never trained on, 55.4% fused - plus 3.3 points over the
best single modality.

LIKELY QUESTION: "Only 55%?" Answer: seven-class cross-dataset, chance
is 14.3%, and the published range for this exact setting is 40-55%. The
90% numbers in the literature are same-dataset. I have a slide on why.

---

## Slide 9 - System Architecture

SYSTEM ARCHITECTURE (45s)

Walk the diagram left to right once, then stop.

Three independent encoders, one shared label space, one
reliability-weighted pool. Note what is NOT here: there is no joint
encoder, no cross-attention between modalities, no learned fusion
layer. Every box can be deleted and the system still answers, with the
remaining signals, and tells you it is doing so.

That is a deliberate trade. We give up whatever a jointly trained model
would gain, and in exchange we get a system that degrades predictably
and can explain any individual decision - which for a clinical tool is
the better side of the trade.

---

## Slide 10 - User Flow

USER FLOW (45s)

This is the therapist's path, not the model's.

They open the portal, start or upload a consultation, and the system
streams back a reading per frame with the weights that produced it.
Nothing is stored that is not needed for the session.

One thing to point at: the output is never just a label. It is a label,
a confidence, three per-modality opinions, and a conflict flag. A
clinician who cannot see why a system said "sad" will not use it twice.

---

## Slide 11 - Technology Stack  (new)

TECHNOLOGY STACK (75s - new slide)

Three columns: what trains it, what serves it, what shows it. Read the
labels, not the whole card - the detail is there for the panel to scan.

Train: PyTorch and HuggingFace Transformers for all three encoders, and
the thing worth saying out loud is the last line - all of this trained
on one RTX 4060. No cluster, no cloud credits.

Serve: FastAPI and Uvicorn, and ONE WebSocket carrying video frames,
sixteen-kilohertz audio and typed text on a single connection. The face
model runs through ONNX Runtime on the GPU and falls back to CPU by
itself. The fusion is about forty lines of NumPy - and every constant in
it lives in a JSON file rather than in code, so a weight can be retuned
and cited without a commit to the engine.

Show: the client is one HTML file. No React, no build step. The detail I
would point at is pacing - the browser only sends the next frame when
the server acknowledges the last one, so a slow machine loses frame rate
instead of building a queue of stale frames.

The bottom line is the argument: a telemedicine deployment cannot assume
a datacentre. Everything degrades to CPU on one machine, and the
heaviest piece - a 380 MB voice model - is opt-in, so a session that
never uses the microphone never loads it.

LIKELY QUESTION: "Why ONNX for one model and PyTorch for the others?"
The face model runs thirty times a second; the other two about once a
second. The ViT was worth exporting; the others were not, and keeping
them in PyTorch means we can swap checkpoints without re-exporting.

---

## Slide 12 - Why Three Modalities

WHY THREE MODALITIES (75s)

The argument is not "more signals are better". It is that these three
fail in DIFFERENT situations.

Face: always present, always current - and the weakest cross-dataset.
74% on its own test split, 52% on faces it has never seen. That
22-point drop is the single most important number in this deck. A still
face is also genuinely ambiguous; humans disagree about it too.

Voice: carries intensity the face hides. A flat expression with a sharp
voice is still distress. 69% on clear acted speech, 36% in a noisy
room, and nothing at all when nobody is talking.

Text: the most explicit signal when it exists - someone typing "I can't
cope" is not ambiguous - but it arrives in bursts and goes stale. A
message from two minutes ago says very little about right now, which is
why text evidence decays with a thirty-second half-life.

The consequence: because the three fail independently, the pool can
recover when one is wrong. And because each carries a confidence, the
fusion knows when to stop trusting it rather than being switched off by
hand.

---

## Slide 13 - Meet the Three Encoders  (new)

MEET THE THREE ENCODERS (2 min - new slide; this is the "explain it"
slide, so take your time and use the analogies)

Each card is the same three questions: what is it, why does it suit
this signal, what does it cost us.

THE FACE - ViT-Base/16. A Vision Transformer cuts the image into a grid
of sixteen-by-sixteen pixel patches, treats each patch like a word in a
sentence, and lets every patch attend to every other patch at once.

Why that suits a face: the difference between angry and sad is not in
the brow alone or the mouth alone - it is in how they move TOGETHER. A
transformer relates two distant regions in a single step. A convolution
with a small kernel has to stack layer on layer before the brow and the
mouth are even in the same receptive field.

Cost: 86 million parameters, exported to ONNX, 6.6 milliseconds a frame.

THE VOICE - WavLM-base-plus. This is the one worth slowing down on,
because the interesting part is how it was trained BEFORE we touched
it. Microsoft trained it on roughly ninety-four thousand hours of
unlabelled speech by masking out pieces of the audio and asking the
model to reconstruct what was missing. No emotion labels at all at that
stage - it is learning what human speech is like.

Why that matters to us: by the time we fine-tune it, it already knows
what voices do - pitch movement, timing, hesitation, vocal strain. Our
labelled dataset is 4,179 clips. You cannot learn "what speech sounds
like" from four thousand clips. You can only learn to map something you
already understand onto seven labels, which is exactly what fine-tuning
does.

Analogy if the panel wants one: it is the difference between teaching
someone to read emotion in a language they already speak, versus
teaching them the language and the emotion at the same time from four
thousand sentences.

Cost: 95 million parameters, 16 milliseconds for a four-second window,
scored about once a second.

THE WORDS - TinyBERT, four layers. Distillation: you take a full-size
BERT, and you train a small four-layer student to reproduce the big
model's answers rather than learning from the labels alone. The student
inherits a lot of the teacher's behaviour at a fraction of the size -
14.4 million parameters, about an eighth of BERT-base.

Why that suits text here: text is the lightest of our three signals and
must not take the compute budget. At 2.8 milliseconds a message, it runs
on the CPU beside two other models without being noticed.

Cost: we deploy the GoEmotions checkpoint specifically, because it
predicts our seven classes directly.

THE BOTTOM LINE - and this is the slide's real point: all three must
output the same seven emotions in the same order, so fusing them is
arithmetic rather than translation. That constraint, not accuracy, is
what ruled several otherwise better models out - which is the next
slide.

---

## Slide 14 - Why These, Not the Alternatives  (new)

WHY THESE, NOT THE ALTERNATIVES (2 min - new slide)

This table is the defence. Every row is: what we chose, what we measured
it against, what the measurement said. Read the rows, do not read the
cells word for word.

FACE. We chose our v4 ViT checkpoint, and we measured it against three
of our own attempts to beat it: balanced sampling, label cleaning, and
an ensemble of two checkpoints. All three scored worse on unseen faces.
The clearest case is label cleaning: removing 4,614 confidently
mislabelled images raised validation accuracy from 77.5 to 81.1 percent,
and dropped unseen-face accuracy from 52.3 to 48.0. What it removed was
disproportionately the AMBIGUOUS examples, and those are what teach a
model to generalise. The conclusion we draw is important: our binding
constraint is the diversity of the face data, not the model. A bigger
backbone does not fix this.

VOICE. WavLM against this project's own MFCC-CNN - and we did not just
cite a paper, we imported that architecture unchanged and retrained it
on the identical speaker-disjoint split. Macro-F1 went from 0.322 to
0.499. Because the data and splits are held constant, that 17.7-point
gap is attributable to the model and nothing else.

TEXT. TinyBERT against DistilBERT and MobileBERT on the same split:
88.6 macro-F1 against 89.3 and 89.4. We give up 0.8 points of F1 and
get 2.8 milliseconds instead of 10.6 and 16.9. For three models sharing
a CPU, that is the right side of the trade. BERT-base and RoBERTa were
excluded on budget, not on accuracy.

TEXT HEAD. This row is the most instructive. We deploy the GoEmotions
checkpoint even though the MTEB checkpoint scores higher - because MTEB
has six classes and physically cannot say "neutral" or "disgust".
Neutral is the most common state in a real consultation. A better score
on a label set you cannot use is worth nothing.

FUSION. A log opinion pool rather than a learned fusion head, because
no trimodal corpus exists in our label space. A learned head would have
to be fitted on simulated pairs, and it could not be audited.

THEN THE FOOTER, deliberately: this is where slides 6 and 7 are
reconciled. Say it plainly - "Combination A proposed MentalBERT, WavLM
and DINOv2. WavLM survived. The other two are not in the codebase:
neither predicts our seven classes directly and both exceed the CPU
budget. The F1 figures on slide 7 belong to that earlier study, not to
this system."

Saying that yourself converts your weakest slide into evidence that you
audit your own work.

---

## Slide 15 - Rebuilding the Audio Corpus

REBUILDING THE AUDIO CORPUS (75s)

This slide is about a problem we inherited and had to fix before any
audio number meant anything.

The earlier branch shipped an audio model that had never been trained -
the notebook stopped before the training cell - on a corpus missing two
of the seven emotions. No neutral and no disgust, and an 8.6-to-1
imbalance on surprise. A model that cannot emit "neutral" will label
silence as whatever its prior favours, on every single frame. Neutral is
the resting state of a real consultation, so that is fatal, not
cosmetic.

So we rebuilt it from four corpora - CREMA-D, RAVDESS, SAVEE and MELD -
about 23,000 clips covering all seven classes.

The methodological point is the caption, and I want to dwell on it. We
split by SPEAKER, not by clip. Every acted corpus has each actor perform
every emotion on the same sentences. If you split randomly, the same
actor's voice appears in train and test, and the model can score by
recognising the voice rather than the emotion. That is worth fifteen to
twenty-five points, and it is why a lot of published speech-emotion
numbers look high. Our numbers are lower than the literature partly
because of this choice, and we would make it again.

---

## Slide 16 - Audio Modality - Results

AUDIO RESULTS (75s)

Speaker-disjoint test split, 4,179 clips, seven classes, so chance is
14.3%.

The baseline bar is the earlier branch's own MFCC CNN architecture,
imported unchanged and retrained on the same rebuilt data. Same data,
same splits - so the 17.7-point gap isolates the model from the
dataset. 0.322 macro-F1 to 0.499.

Now the part I would rather you hear from me than find in the appendix.
The averaged number hides the real finding. Broken out by domain: 69.4%
accuracy on acted studio speech, 35.6% on MELD - real dialogue from a TV
show, with background noise and a laugh track. Studio speech works. A
noisy room is still hard.

And the fusion is told the harder number. The per-class reliabilities we
feed the pool come from the wild domain, not the acted one, so the
system does not over-trust the voice in exactly the conditions a real
consultation resembles.

---

## Slide 17 - Does Fusion Actually Help?

DOES FUSION ACTUALLY HELP? (90s)

MELD test split, 2,523 clips that carry both audio and a transcript.
Solid bars are accuracy, hatched are macro-F1.

Yes, it helps: 55.4% fused against 52% for the best single modality -
plus 3.3 points.

Two honest qualifications, and I would rather state them than be asked.

First, why these are real pairs. MELD ships a transcript per clip, so
the audio and the text describe the same utterance from the same
speaker. A lot of multimodal papers pair a face from one dataset with a
voice from another at random, which manufactures independence that does
not exist and inflates the fusion gain.

Second - and this is why our gain is modest - on real data the
modalities fail TOGETHER. A flat delivery tends to come with a flat
sentence. Independently paired data cannot show you that, which is
exactly why it overstates the benefit. Our +3.3 is a smaller number
measured honestly.

LIKELY QUESTION: "Is the face leg real in this measurement?" No, and it
is flagged in multimodal.json - MELD has no usable face crops for our
pipeline, so the face leg is simulated from the measured v4 confusion
behaviour. The audio-text gain is fully real; the trimodal figure
carries that caveat and we do not hide it.

---

## Slide 18 - Scoring the Multimodality

SCORING THE MULTIMODALITY (60s)

We wanted a single number we could argue about rather than a vague claim
that the system works, so this is a composite of five components with
stated weights.

Capability, 0.633 at weight 0.35 - trimodal macro-F1 against a 0.70
ceiling. Fusion gain, 0.222 at 0.25. Calibration, 0.376 at 0.15 - audio
is the best-calibrated leg. Robustness, 0.609 at 0.15 - the worst
two-modality result as a fraction of all three. Conflict discrimination,
0.721 at 0.10 - 74.1% correct when the modalities agree against 52.5%
when they clash.

Total: 49.7 out of 100.

Be explicit about what that is: this project's own summary of "is the
pool worth having", not a standard metric. Every component, its raw
value and its weight is written to results/multimodal.json, so anyone
can change a weight and recompute. We publish it precisely because it is
arguable.

The component worth noticing is conflict discrimination. The gap between
74% and 52.5% means the conflict flag is predictive - when the system
says the modalities disagree, it is genuinely less reliable. That is a
usable warning, not decoration.

---

## Slide 19 - The Live System

THE LIVE SYSTEM (30s - set up the demo)

This is running, not mocked. Face, voice and text fused in real time at
32 frames a second on one laptop GPU, with the reasoning shown alongside
the answer.

If demoing live: smile, then type "i am furious" and let them watch the
weights move. If not, the next two slides walk the interface.

---

## Slide 20 - Interface screenshot

FULL SCREENSHOT (20s)

Give them a moment to take in the whole interface, then say:

"Everything on this screen is produced per frame - the three modality
readings, the weights, the fused answer and the conflict banner. The
next slide pulls out the three parts that matter."

---

## Slide 21 - Inside the Interface

INSIDE THE INTERFACE (90s)

Three things, and they are the design argument of the whole project.

Three readings, nothing hidden. On this frame the face says happy, the
voice says angry, the text says happy. A system that showed only a fused
answer would have swallowed that disagreement.

Weighted, not averaged. Face 10%, audio 26%, text 63% on this frame. The
face crop was low quality - blurred, badly posed - so it lost influence
automatically. Nobody switched it off; the quality term in the exponent
did it. That is the mechanism from the methodology slide doing visible
work.

Disagreement surfaced. When the modalities conflict, the banner says so.
And that flag is predictive, not cosmetic: accuracy drops from 74% to
53% when it fires. So the right reading of a conflict banner is "ask the
patient rather than trust the label".

LIKELY QUESTION: "How do you detect conflict?" Two conditions together -
the modalities' top choices differ AND their Bhattacharyya overlap is
below 0.8. Either alone misfires: a low overlap with the same top choice
is just differing confidence about one emotion, and differing top
choices with high overlap is two near-uniform shrugs.

---

## Slide 22 - The Label Will Not Sit Still  (new)

THE LABEL WILL NOT SIT STILL (2 min - the most important slide; do not
skip it and do not let a panel member find it instead of you)

Frame it before the content: "The system works. It is also not yet
trustworthy in the way a clinical tool has to be, and I want to tell you
exactly why rather than have you discover it in the demo."

Each card is a number, what it means, and the consequence.

ONE - the top-1 is nearly a tie. 52.3% over seven classes means the
leading class and the runner-up are usually within a few points. Any
change in crop, lighting or head pose re-orders them, so the answer
moves while the face does not. The pairs that swap most are neutral
against sad and angry against disgust - and disgust holds only 0.20 F1
cross-dataset, barely better than a guess.

TWO - the session prior is wrong. A webcam consultation is roughly 55%
neutral, but the model is calibrated on a balanced set, and a model
carries whatever prior it was calibrated under. Measured under a
realistic prior it emits neutral on 27.5% of frames, sad at 2.2 times
and surprise at 2.1 times their true rate. So nearly every spurious
"surprise" flash a user sees is a neutral face leaking. Correcting the
prior takes accuracy from 56.7% to 72.3% - and we ship the correction
switched OFF, because turning it on would change what every number in
our evaluation harness means. That is the right default for a research
artefact and the wrong default for a product, and we know which one we
are building next.

THREE - only the face leg is smoothed. There is an exponential moving
average with a 0.6-second time constant and a hysteresis margin inside
the face pipeline. But the fusion engine is stateless by design, so the
number on screen has no temporal filter at all. Worse, its weights move
every frame as audio decays with an eight-second half-life and text with
a thirty-second one - which means the displayed label can change with no
new evidence whatsoever.

Then the bottom line, said plainly: nothing in this deck measures this.
Every figure we have shown is per-frame accuracy on still images. Not
one metric describes how often the displayed label changes, which is the
first thing a clinician would notice. That is why it is item one in the
future scope.

LIKELY QUESTION: "So is the accuracy real?" Yes - per-frame accuracy is
exactly what we measured, honestly, on held-out data. Stability is a
different property that we did not measure, and the fix is a decision
rule, not a better model.

---

## Slide 23 - Future Scope  (filled in)

FUTURE SCOPE (2 min)

Say the organising principle first: this is ordered, not listed. The
instability has to be measured and closed before any clinical claim is
worth making. Each card is a "do" and a "why" - read those two lines,
not the whole card.

NEXT - make the live output trustworthy.

One, measure stability. Switches per minute, mean dwell time, and the
entropy of the displayed sequence, over held-out video, beside accuracy.
A model that is 52% accurate and steady is clinically usable; one that
is 52% accurate and flickering is not - and today our metrics cannot
tell those two apart. This is cheap, and it comes first.

Two, smooth the decision rather than the leg. Move the EMA and
hysteresis AFTER the pool instead of inside the face pipeline, then
replace both with an explicit dwell-time model - a seven-state hidden
Markov model decoded with Viterbi - so changing the reported emotion has
to be paid for by evidence. And turn the prior correction on, reported
as a serving rule so the evaluation numbers stay comparable.

Three, report episodes, not frames. A clinician does not need 32 labels
a second. Aggregate to intervals - "two minutes forty predominantly sad,
two flagged conflicts" - with per-frame output available on request.
That also gives us somewhere to put abstention: on a low-quality or
conflicted stretch the honest output is "unknown", and right now we have
no way to say it.

THEN - what a clinical claim would require.

Four, escape the data ceiling. Our three attempts to beat v4 all failed,
which is evidence that the binding constraint is face diversity, not
capacity. The levers left are the full AffectNet - about 287,000 images
against the 27,823 public subset, which needs an authenticated licence -
and a genuinely trimodal corpus like IEMOCAP or CMU-MOSEI, so fusion can
be fitted and tested on real pairs instead of simulated ones.

Five, validate clinically. Every number in this deck comes from acted
corpora or TV dialogue. The phrase "emotionally intelligent
telemedicine" is not earned until we have agreement against clinician
ratings on real consultations, under ethics approval - with accuracy
reported per skin tone, age and gender, because a face model carries
that fairness risk by default and our training data is not balanced on
any of those axes.

Six, privacy by construction. Right now frames cross the wire as Base64
JPEG, which for a medical consultation is the wrong default. Moving the
face leg into the browser with ONNX Runtime Web keeps pixels on the
device and sends only a seven-number probability vector. Federated
fine-tuning is then the route to clinical data that cannot legally be
centralised - which is reference nine on the next slide.

---

## Slide 24 - References

REFERENCES (15s)

Do not read these. One sentence:

"Ten papers; the ones that shaped the design are four and six on
multimodal fusion, two on real clinical doctor-patient video, and nine
on federated learning, which is where the privacy plan comes from."

---

## Slide 25 - Thank You

CLOSE (30s)

Close on the honest version, not a boast:

"Three transformers, one shared label space, one interpretable pool. The
fused system beats every single modality by 3.3 points measured on real
pairs with speaker-disjoint splits, and it tells you which signal drove
each decision. It is also not yet stable enough for a clinician, we know
precisely why, and fixing that is a decision rule rather than a new
model.

Thank you - happy to take questions."

PREPARED ANSWERS:
- "Why is accuracy so low?" Seven-class, cross-dataset, chance 14.3%,
  published range 40-55%. Same-dataset numbers are not comparable, and
  our own FER2013 split had 7.9% train-test image overlap which we found
  and removed.
- "Why not a single end-to-end multimodal model?" No trimodal corpus in
  this label space, and we would lose the per-modality audit trail.
- "Slide 7 says Combination A, but you built something else." Correct,
  and slide 14 is the reconciliation. WavLM carried over; MentalBERT and
  DINOv2 did not, on label space and CPU budget.
- "What is the single biggest weakness?" Temporal stability, measured by
  nothing today. Slide 22.
- "What would you do with another month?" Items one to three of the
  future scope, in that order.

---
