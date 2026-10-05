# Sept eval deck

`Sept_eval_1_multimodal.pptx` — 24 slides, with the speaker script
embedded as PowerPoint notes on every slide and duplicated in
`PRESENTATION_SCRIPT.md` for printing.

## What changed from the uploaded version

| Slide | Change |
|---|---|
| 11 | **Technology Stack** — new. Train / serve / show, and why each choice is sized for one machine. |
| 13 | **Why These Models** — new. Each encoder against the alternative it was measured against. |
| 21 | **The Label Will Not Sit Still** — new. The three measured causes of the live output fluctuating on a steady face. |
| 22 | **Future Scope** — was an empty title. Six items in two ordered groups. |
| all | Speaker notes added. |

Nothing else in the original 21 slides was rewritten. Slides 6 and 7 are
blank in the source deck and are still blank — delete them or put the
architecture figure in one.

## Where the numbers come from

Every figure quoted on the new slides is in a committed result file on
`multimodal-v7-audio`, and the speaker notes name the file so it can be
opened mid-question:

| Claim | File |
|---|---|
| WavLM 0.499 vs MFCC CNN 0.322 macro-F1; 69.4% acted / 35.6% wild; latencies | `audio-emotion-module/results/benchmark.json` |
| 55.4% fused, +3.3 over best single modality; composite 49.7/100 | `audio-emotion-module/results/multimodal.json` |
| TinyBERT 88.6 vs DistilBERT 89.3 vs MobileBERT 89.4 macro-F1; 2.8/10.6/16.9 ms | `text-emotion-module/results/tables.md` |
| Face 52.3% / 49.7% macro-F1 on the untouched unseen split | `vit-emotion-v2-deployment/results_face_eval.v4.unseen_test.json` |
| neutral shown 27.5% vs 55% true; sad 2.2x, surprise 2.1x; 56.7% → 72.3% | `vit-emotion-v2-deployment/results_live_prior.json` |
| EMA τ=0.6 s, hysteresis 0.05, audio/text half-lives 8 s / 30 s | `vit-emotion-v2-deployment/face_pipeline.py`, `reliability.json` |
| v5 / v6 / ensemble all worse than v4 | `training/README.md` |

## Rebuilding

```bash
python build_deck.py <uploaded.pptx> built.pptx
python add_notes.py built.pptx Sept_eval_1_multimodal.pptx PRESENTATION_SCRIPT.md
```

`build_deck.py` adds the three new slides and fills Future Scope;
`add_notes.py` attaches `speaker_notes.py` and writes the printable
script. Edit the script in `speaker_notes.py`, never in the `.pptx` — a
rebuild overwrites the notes.

The deck is hand-authored in Canva and has no usable slide layouts:
every slide is `Blank` with its furniture drawn as four freeform groups.
New slides therefore clone that furniture out of the existing Future
Scope slide and write their bodies in the same XML vocabulary the other
content slides use (tokens at the top of `build_deck.py`).

LibreOffice cannot render in the build container, so card heights are
checked arithmetically rather than visually: `build_deck.py` estimates
each text block's height with deliberately pessimistic font metrics and
fails the build if any content crosses the 10.10 in decoration line.
Open the deck once in PowerPoint before presenting.
