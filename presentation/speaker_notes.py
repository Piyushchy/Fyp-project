"""
Speaker notes for Sept_eval_1_multimodal.pptx, keyed by final slide number.

Written to be spoken, not read off the slide. Each entry opens with the
one sentence that has to land, then the supporting figures, then the
question the panel is most likely to ask and the answer to it.

Every number appears in a committed result file; the file is named in
the note so it can be opened mid-question.
"""

NOTES: dict[int, str] = {}


NOTES[1] = """
OPEN (20s)

Good morning. We are building an emotion recognition layer for
telemedicine - a system that watches a consultation and tells the
clinician what the patient's face, voice and words suggest they are
feeling, in real time.

I am <name>, with Muskan, Yashraj and Ankita, mentored by Dr. Girishma
Sharma.

One framing note before I start. Since the last review we have gone from
one modality to three, and the honest headline is that the fused system
is better than any single signal and still not stable enough to put in
front of a clinician. I will show you both halves of that, with numbers.
"""


NOTES[2] = """
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

Keep one thing in mind for later - "in real time" is the hard part, and
it is where our remaining problem lives.
"""


NOTES[3] = """
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
"""


NOTES[4] = """
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

Finalise the web platform UI. Done - the therapist-facing portal is
working, and it shows not just the answer but which modality produced
it.
"""


NOTES[5] = """
METHODOLOGY - SECTION DIVIDER (10s)

Just a divider. Say the sentence and move on:

"The method is three transformers that never talk to each other, and
one piece of arithmetic that combines them."

[Note for the team: slides 6 and 7 are blank. Either delete them before
the review or drop the architecture figure into one of them - a blank
slide mid-section reads as an accident.]
"""


NOTES[6] = """
BLANK SLIDE - delete before presenting, or move the architecture figure
here. If it is still in the deck on the day, press Next immediately and
say nothing.
"""


NOTES[7] = """
BLANK SLIDE - same as the previous one. Delete or fill.
"""


NOTES[8] = """
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
arithmetic rather than translation - position three means happy in all
three vectors. Getting the audio model to speak those seven directly
took rebuilding its training corpus, which is the slide after next.

Step three, weigh and decide. A weighted logarithmic opinion pool - a
geometric mean, not an average. Geometric because the modalities are
roughly independent given the true emotion, so a class both find
plausible should be reinforced and a class either rules out should be
suppressed. An arithmetic average can never be more confident than its
inputs; a product of experts can.

And the weights are not fixed. They track how much THIS observation
deserves trust: face crop quality, how recently the voice was heard,
how long the typed message was.

The numbers on the right: 52% face, 50% voice, 33% text measured
alone on data they never trained on, 55.4% fused - plus 3.3 points
over the best single modality.

LIKELY QUESTION: "Only 55%?" Answer: seven-class cross-dataset, chance
is 14.3%, and the published range for this exact setting is 40-55%. The
90% numbers in the literature are same-dataset. I have a slide on why.
"""


NOTES[9] = """
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
"""


NOTES[10] = """
USER FLOW (45s)

This is the therapist's path, not the model's.

They open the portal, start or upload a consultation, and the system
streams back a reading per frame with the weights that produced it.
Nothing is stored that is not needed for the session.

One thing to point at: the output is never just a label. It is a
label, a confidence, three per-modality opinions, and a conflict flag.
A clinician who cannot see why a system said "sad" will not use it
twice.
"""


NOTES[11] = """
TECHNOLOGY STACK (75s - new slide)

Three columns: what trains, what serves, what shows.

Train: PyTorch with HuggingFace Transformers for all three encoders.
The face model is fine-tuned from google/vit-base-patch16-224-in21k
over four merged face corpora. WavLM-base-plus and TinyBERT fine-tune
in the same framework. librosa and soundfile condition the audio,
scikit-learn computes every metric. All of the training ran on one
RTX 4060 - no cluster, no cloud credits.

Serve: FastAPI and Uvicorn, with one WebSocket carrying video frames,
sixteen-kilohertz PCM and typed text on a single connection. The face
ViT runs through ONNX Runtime on the CUDA provider and falls back to
CPU automatically. WavLM and TinyBERT stay in PyTorch. The fusion pool
itself is about forty lines of NumPy. Face detection and the five
landmarks for alignment come from OpenCV's YuNet.

Show: the client is a single HTML file. No React, no build step.
getUserMedia and a canvas for video, an AudioWorklet for audio, and
back-pressure - the browser only sends the next frame when the server
acknowledges the last, so a slow machine drops frame rate instead of
building a queue.

The point of the bottom line: a telemedicine deployment cannot assume a
datacentre. Everything here degrades to CPU on one machine, and the
heaviest piece - a 380 MB WavLM checkpoint - is opt-in, so a session
that never uses the microphone never loads it.

LIKELY QUESTION: "Why ONNX for one model and PyTorch for the other
two?" Because the face model runs 30 times a second and the other two
about once a second. The ViT was worth exporting and quantising; the
others were not, and keeping them in PyTorch means we can swap
checkpoints without re-exporting.
"""


NOTES[12] = """
WHY THREE MODALITIES (75s)

The argument is not "more signals are better". It is that these three
fail in different situations.

Face: always present, always current - and the weakest cross-dataset.
74% on its own test split, 52% on faces it has never seen. That
22-point drop is the single most important number in this deck. A still
face is also genuinely ambiguous; humans disagree about it too.

Voice: carries intensity the face hides. A flat expression with a sharp
voice is still distress. 69% on clear acted speech, 36% in a noisy
room, and nothing at all when nobody is talking.

Text: the most explicit signal when it exists - someone typing "I
can't cope" is not ambiguous - but it arrives in bursts and goes stale.
A message from two minutes ago says very little about right now, which
is why text evidence decays with a thirty-second half-life.

The consequence: because the three fail independently, the pool can
recover when one is wrong. And because each carries a confidence, the
fusion knows when to stop trusting it rather than being switched off by
hand.
"""


NOTES[13] = """
WHY THESE MODELS (2 min - new slide, expect questions here)

Each of these was chosen against a measured alternative on the same
data. None of it came off a leaderboard.

Face - ViT-Base/16. Why a transformer and not a CNN: angry and sad
differ in how the brow and the mouth move TOGETHER, and self-attention
over sixteen-by-sixteen patches relates two distant regions in a single
layer, where a small-kernel CNN needs depth to reach across the face.
The more important point is the second half of that card: capacity is
not our binding constraint. We tried three things to beat our v4
checkpoint - balanced sampling, label cleaning, and a two-checkpoint
ensemble - and all three measured WORSE on unseen faces. Label cleaning
raised validation accuracy from 77.5% to 81.1% and dropped
unseen-face accuracy from 52.3% to 48.0%. What it removed was the
ambiguous examples, and those are what teach generalisation. So the
lever is data diversity, not a bigger backbone.

Voice - WavLM-base-plus. We benchmarked it against this project's own
MFCC CNN, retrained unchanged on the identical speaker-disjoint split.
Macro-F1 went from 0.322 to 0.499 - a 17.7-point gap attributable to
the model alone, because the data and splits were held constant.
Self-supervised pretraining on roughly 94,000 hours of speech is
knowledge our 4,179 clips cannot supply. One detail worth knowing:
we learn a softmax mixture over all thirteen hidden states, and it
settles on layers nine and ten, not the top. The masked-prediction
objective pulls the final layers toward phonetic content, so using
the last layer alone would literally discard the emotion.

Text - TinyBERT, four layers, 312 hidden. Benchmarked against
DistilBERT at 67 million parameters and MobileBERT at 24.6 million on
the same split: 88.6 macro-F1 against 89.3 and 89.4. So we give up
0.8 points and get 2.8 milliseconds instead of 10.6 and 16.9, and 7.6
times smaller than BERT-base. BERT-base and RoBERTa were excluded on
the CPU budget, not on accuracy. And the checkpoint we deploy is the
GoEmotions one rather than the better-scoring MTEB one, because
GoEmotions predicts the shared seven directly - the six-class MTEB
head physically cannot say "neutral" or "disgust", and neutral is the
most common state in a real session.

Bottom line - why fusion is not a fourth model: a learned fusion head
needs trimodal training data in this label space and none exists. It
would have to be fitted on simulated pairs, and it could not be
audited. The log pool has no parameters to fit, survives a missing
modality, and reports the weight it gave each signal - which is the
thing a clinician actually has to check.
"""


NOTES[14] = """
REBUILDING THE AUDIO CORPUS (75s)

This slide is about a problem we inherited and had to fix before any
audio number meant anything.

The earlier branch shipped an audio model that had never been trained -
the notebook stopped before the training cell - on a corpus missing two
of the seven emotions. No neutral and no disgust, and an 8.6-to-1
imbalance on surprise. A model that cannot emit "neutral" will label
silence as whatever its prior favours, on every single frame. Neutral
is the resting state of a real consultation, so that is fatal, not
cosmetic.

So we rebuilt it from four corpora - CREMA-D, RAVDESS, SAVEE and MELD -
about 23,000 clips covering all seven classes.

The methodological point is the caption, and I want to dwell on it. We
split by SPEAKER, not by clip. Every acted corpus has each actor
perform every emotion on the same sentences. If you split randomly, the
same actor's voice appears in train and test, and the model can score
by recognising the voice rather than the emotion. That is worth fifteen
to twenty-five points, and it is why a lot of published speech-emotion
numbers look high. Our numbers are lower than the literature partly
because of this choice, and we would make it again.
"""


NOTES[15] = """
AUDIO RESULTS (75s)

Speaker-disjoint test split, 4,179 clips, seven classes, so chance is
14.3%.

The baseline bar is the earlier branch's own MFCC CNN architecture,
imported unchanged and retrained on the same rebuilt data. Same data,
same splits - so the 17.7-point gap isolates the model from the
dataset. 0.322 macro-F1 to 0.499.

Now the part I would rather you hear from me than find in the appendix.
The averaged number hides the real finding. Broken out by domain:
69.4% accuracy on acted studio speech, 35.6% on MELD - real dialogue
from a TV show with background noise and a laugh track. Studio speech
works. A noisy room is still hard.

And the fusion is told the harder number. The per-class reliabilities
we feed the pool come from the wild domain, not the acted one, so the
system does not over-trust the voice in exactly the conditions a real
consultation resembles.
"""


NOTES[16] = """
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

LIKELY QUESTION: "Is the face leg real in this measurement?" No, and
it is flagged in multimodal.json - MELD has no usable face crops for
our pipeline, so the face leg is simulated from the measured v4
confusion behaviour. The audio-text gain is fully real; the trimodal
figure carries that caveat and we do not hide it.
"""


NOTES[17] = """
SCORING THE MULTIMODALITY (60s)

We wanted a single number we could argue about rather than a vague
claim that the system works, so this is a composite of five components
with stated weights.

Capability, 0.633 at weight 0.35 - trimodal macro-F1 against a 0.70
ceiling. Fusion gain, 0.222 at 0.25. Calibration, 0.376 at 0.15 - and
audio is the best-calibrated leg. Robustness, 0.609 at 0.15 - the
worst two-modality result as a fraction of all three. Conflict
discrimination, 0.721 at 0.10 - 74.1% correct when the modalities
agree against 52.5% when they clash.

Total: 49.7 out of 100.

Be explicit about what that is: it is this project's own summary of
"is the pool worth having", not a standard metric. Every component,
its raw value and its weight is written to results/multimodal.json, so
anyone can change a weight and recompute. We publish it precisely
because it is arguable.

The one component worth noticing is conflict discrimination. The gap
between 74% and 52.5% means the conflict flag is predictive - when the
system says the modalities disagree, it is genuinely less reliable.
That is a usable warning, not decoration.
"""


NOTES[18] = """
THE LIVE SYSTEM (30s - set up the demo)

This is running, not mocked. Face, voice and text fused in real time at
32 frames a second on one laptop GPU, with the reasoning shown
alongside the answer.

If demoing live: smile, then type "i am furious" and let them watch the
weights move. If not, the next two slides walk the interface.
"""


NOTES[19] = """
FULL SCREENSHOT (20s)

Give them a moment to take in the whole interface, then say:

"Everything on this screen is produced per frame - the three modality
readings, the weights, the fused answer and the conflict banner. The
next slide pulls out the three parts that matter."
"""


NOTES[20] = """
INSIDE THE INTERFACE (90s)

Three things, and they are the design argument of the whole project.

Three readings, nothing hidden. On this frame the face says happy, the
voice says angry, the text says happy. A system that showed only a
fused answer would have swallowed that disagreement.

Weighted, not averaged. Face 10%, audio 26%, text 63% on this frame.
The face crop was low quality - blurred, badly posed - so it lost
influence automatically. Nobody switched it off; the quality term in
the exponent did it. That is the mechanism from the methodology slide
doing visible work.

Disagreement surfaced. When the modalities conflict, the banner says
so. And that flag is predictive, not cosmetic: accuracy drops from 74%
to 53% when it fires. So the right reading of a conflict banner is "ask
the patient rather than trust the label".

LIKELY QUESTION: "How do you detect conflict?" Two conditions
together - the modalities' top choices differ AND their Bhattacharyya
overlap is below 0.8. Either alone misfires: a low overlap with the
same top choice is just differing confidence about one emotion, and
differing top choices with high overlap is two near-uniform shrugs.
"""


NOTES[21] = """
THE LABEL WILL NOT SIT STILL (2 min - the most important slide; do not
skip it and do not let it be found by a panel member instead)

Frame it before the content: "The system works. It is also not yet
trustworthy in the way a clinical tool has to be, and I want to tell
you exactly why rather than have you discover it in the demo."

The symptom: hold a steady expression and the displayed emotion still
changes several times a second. Three measured causes.

One, the top-1 is nearly a tie. 52.3% over seven classes means the
leading class and the runner-up are usually within a few points. Any
change in crop, lighting or head pose re-orders them, so the argmax
moves while the face does not. The pairs that swap most are neutral
against sad and angry against disgust - and disgust holds only 0.20 F1
cross-dataset, so it is barely better than a guess.

Two, the session prior is wrong. A webcam consultation is roughly 55%
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

Three, only the face leg is smoothed. There is an exponential moving
average with a 0.6-second time constant and a hysteresis margin inside
the face pipeline. But the fusion engine is stateless by design, so the
number on screen has no temporal filter at all. Worse, its weights move
every frame as audio decays with an eight-second half-life and text
with a thirty-second one - which means the displayed label can change
with no new evidence whatsoever.

Then the bottom line, and say it plainly: nothing in this deck measures
this. Every figure we have shown is per-frame accuracy on still images.
Not one metric describes how often the displayed label changes, which
is the first thing a clinician would notice. That is why it is item one
in the future scope.

LIKELY QUESTION: "So is the accuracy real?" Yes - per-frame accuracy is
exactly what we measured, honestly, on held-out data. Stability is a
different property that we did not measure, and the fix is a decision
rule, not a better model.
"""


NOTES[22] = """
FUTURE SCOPE (2 min)

Say the organising principle first: this is ordered, not listed. The
instability has to be measured and closed before any clinical claim is
worth making.

NEXT - make the live output trustworthy.

One, measure stability. Switches per minute, mean dwell time per label,
and the entropy of the displayed sequence, over held-out video, beside
accuracy. A model that is 52% accurate and steady is clinically usable;
one that is 52% accurate and flickering is not - and today our metrics
cannot tell those two apart. This is cheap and it comes first.

Two, smooth the decision rather than the leg. Move the EMA and
hysteresis AFTER the pool instead of inside the face pipeline, then
replace both with an explicit dwell-time model - a seven-state hidden
Markov model decoded with Viterbi - so changing the reported emotion
has to be paid for by evidence. And enable the prior correction,
reported as a serving rule so the evaluation numbers stay comparable.

Three, report episodes, not frames. A clinician does not need 32 labels
a second. Aggregate to intervals - "two minutes forty predominantly
sad, two flagged conflicts" - with per-frame output available on
request. That also gives us somewhere to put abstention: on a
low-quality or conflicted stretch the honest output is "unknown", and
right now we have no way to say it.

THEN - what a clinical claim would require.

Four, escape the data ceiling. Our three attempts to beat v4 all
failed, which is evidence that the binding constraint is face
diversity, not capacity. The levers left are the full AffectNet - about
287,000 images against the 27,823 public subset, which needs an
authenticated licence - and a genuinely trimodal corpus like IEMOCAP or
CMU-MOSEI, so fusion can be fitted and tested on real pairs instead of
simulated ones.

Five, validate clinically. Every number in this deck comes from acted
corpora or TV dialogue. The phrase "emotionally intelligent
telemedicine" is not earned until we have agreement against clinician
ratings on real consultations, under ethics approval - with accuracy
reported per skin tone, age and gender, because a face model carries
that fairness risk by default and our training data is not balanced on
any of those axes.

Six, privacy by construction. Right now frames cross the wire as
Base64 JPEG, which for a medical consultation is the wrong default.
Moving the face leg into the browser with ONNX Runtime Web keeps pixels
on the device and sends only a seven-number probability vector.
Federated fine-tuning is then the route to clinical data that cannot
legally be centralised - which is reference nine on the next slide.
"""


NOTES[23] = """
REFERENCES (15s)

Do not read these. One sentence:

"Ten papers; the ones that shaped the design are four and six on
multimodal fusion, two on real clinical doctor-patient video, and nine
on federated learning, which is where the privacy plan comes from."
"""


NOTES[24] = """
CLOSE (30s)

Close on the honest version, not a boast:

"Three transformers, one shared label space, one interpretable pool.
The fused system beats every single modality by 3.3 points measured on
real pairs with speaker-disjoint splits, and it tells you which signal
drove each decision. It is also not yet stable enough for a clinician,
we know precisely why, and fixing that is a decision rule rather than a
new model.

Thank you - happy to take questions."

PREPARED ANSWERS:
- "Why is accuracy so low?" Seven-class, cross-dataset, chance 14.3%,
  published range 40-55%. Same-dataset numbers are not comparable, and
  our own FER2013 split had 7.9% train-test image overlap which we
  found and removed.
- "Why not a single end-to-end multimodal model?" No trimodal corpus in
  this label space, and we would lose the per-modality audit trail.
- "What is the single biggest weakness?" Temporal stability, measured
  by nothing today. Slide 21.
- "What would you do with another month?" Items one to three of the
  future scope, in that order.
"""


def as_markdown() -> str:
    """The same notes as a printable script."""

    titles = {
        1: "Title",
        2: "Introduction",
        3: "Motivation",
        4: "Objectives",
        5: "Methodology (divider)",
        6: "[blank slide]",
        7: "[blank slide]",
        8: "Methodology",
        9: "System Architecture",
        10: "User Flow",
        11: "Technology Stack  (new)",
        12: "Why Three Modalities",
        13: "Why These Models  (new)",
        14: "Rebuilding the Audio Corpus",
        15: "Audio Modality - Results",
        16: "Does Fusion Actually Help?",
        17: "Scoring the Multimodality",
        18: "The Live System",
        19: "Interface screenshot",
        20: "Inside the Interface",
        21: "The Label Will Not Sit Still  (new)",
        22: "Future Scope  (filled in)",
        23: "References",
        24: "Thank You",
    }

    lines = [
        "# Presentation script - Sept eval, multimodal emotion recognition",
        "",
        "24 slides, about 22 minutes spoken, which leaves room for questions "
        "in a 30-minute slot.",
        "The same text is embedded as PowerPoint speaker notes, so it is "
        "visible in Presenter View.",
        "",
        "Timing budget: slides 8, 13, 21 and 22 are the long ones (2 min "
        "each). Everything else is 20-90 seconds.",
        "",
        "---",
        "",
    ]

    for number in sorted(NOTES):
        lines.append(f"## Slide {number} - {titles[number]}")
        lines.append("")
        lines.append(NOTES[number].strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
