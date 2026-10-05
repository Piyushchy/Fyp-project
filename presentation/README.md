# Sept eval deck

`Sept_eval_1_multimodal.pptx` — 25 slides, with the speaker script
embedded as PowerPoint notes on every slide and duplicated in
`PRESENTATION_SCRIPT.md` for printing.

## What changed from the uploaded version

| Slide | Change |
|---|---|
| 11 | **Technology Stack** — new. Train / serve / show, as short labelled lines. |
| 13 | **Meet the Three Encoders** — new. What each model *is*, in plain language, and the one property that suits its signal. |
| 14 | **Why These, Not the Alternatives** — new. A comparison grid: what we chose, what we measured it against, what the measurement said. Also reconciles slides 6–7. |
| 22 | **The Label Will Not Sit Still** — new. The three measured causes of the live output fluctuating on a steady face. |
| 23 | **Future Scope** — was an empty title. Six items in two ordered groups. |
| all | Speaker notes added. |

Nothing else was touched. The build verifies this: every one of the
original 21 slides is compared shape-for-shape and run-for-run against
the source, and all 25 media parts are checked byte-identical.

## Slides 6 and 7 need a decision before you present

They are not blank — each is a full-slide image. Slide 6 is the **Models
Used** table (Baseline / Combination A / Combination B), slide 7 is **Why
Combination A Is Superior**.

The problem is that they describe an earlier literature study, and only
one of its three findings reached the built system:

| Slot | Combination A proposed | What actually ships |
|---|---|---|
| Text | MentalBERT | TinyBERT 4L-312D (GoEmotions) |
| Audio | WavLM Base+ | **WavLM-base+** — carried over |
| Video | DINOv2 | ViT-Base/16 |

MentalBERT, DINOv2, MiniLM, MobileNetV3 and EfficientNet appear nowhere
in this repository. Slide 7's figures (fear F1 0.96) are also an order of
magnitude above what the audio module measures on speaker-disjoint
splits, so a panel comparing slide 7 with slide 16 will read it as
inflation unless the gap is named first.

Three options, in order of preference:

1. Keep both slides and say the bridge out loud — the notes for slides 6
   and 7 give the wording, and slide 14's footer closes the loop.
2. Re-title slide 6 "Encoder study — earlier cycle" so the status is
   visible without narration.
3. Remove both slides and let slide 14 carry the model argument alone.

## Where the numbers come from

Every figure on the new slides is in a committed result file on
`multimodal-v7-audio`, and the speaker notes name the file so it can be
opened mid-question:

| Claim | File |
|---|---|
| WavLM 0.499 vs MFCC-CNN 0.322 macro-F1; 69.4% acted / 35.6% wild; latencies | `audio-emotion-module/results/benchmark.json` |
| 55.4% fused, +3.3 over best single modality; composite 49.7/100 | `audio-emotion-module/results/multimodal.json` |
| TinyBERT 88.6 vs DistilBERT 89.3 vs MobileBERT 89.4; 2.8/10.6/16.9 ms | `text-emotion-module/results/tables.md` |
| Face 52.3% / 49.7% macro-F1 on the untouched unseen split | `vit-emotion-v2-deployment/results_face_eval.v4.unseen_test.json` |
| neutral shown 27.5% vs 55% true; sad 2.2x, surprise 2.1x; 56.7% → 72.3% | `vit-emotion-v2-deployment/results_live_prior.json` |
| EMA τ=0.6 s, hysteresis 0.05, audio/text half-lives 8 s / 30 s | `vit-emotion-v2-deployment/face_pipeline.py`, `reliability.json` |
| v5 / v6 / ensemble all worse than v4; v6 81.1% val vs 48.0% unseen | `training/README.md` |

## Rebuilding

```bash
python build_deck.py <uploaded.pptx> built.pptx
python add_notes.py built.pptx Sept_eval_1_multimodal.pptx PRESENTATION_SCRIPT.md
```

Edit the script in `speaker_notes.py`, never in the `.pptx` — a rebuild
overwrites the notes.

The deck is hand-authored in Canva and has no usable slide layouts:
every slide is `Blank` with its furniture drawn as four freeform groups.
New slides therefore clone that furniture out of the existing Future
Scope slide and write their bodies in the same XML vocabulary the other
content slides use (design tokens at the top of `build_deck.py`).

LibreOffice cannot render in the build container, so box heights are
checked arithmetically rather than visually: `build_deck.py` estimates
each text block's height with deliberately pessimistic font metrics and
fails the build if any content crosses the 10.10 in decoration line.
Open the deck once in PowerPoint before presenting.
