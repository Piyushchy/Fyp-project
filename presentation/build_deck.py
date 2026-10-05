"""
Rebuild Sept_eval_1_multimodal.pptx with the slides the September review
left open.

Added, in presentation order:

  11  Technology Stack              what it is built from
  13  Meet the Three Encoders       what each model IS, in plain language
  14  Why These, Not the Others     what each was measured against
  22  The Label Will Not Sit Still  the live output's real weakness

and Future Scope, which was an empty title, plus speaker notes on every
slide (added separately by add_notes.py).

The deck is hand-authored in Canva and has no usable layouts - every
slide is 'Blank' with its decoration drawn as four freeform groups. So
new slides are built by cloning the decoration and title scaffold out of
the existing Future Scope slide and writing the body in the same XML
vocabulary the other content slides use. Nothing in the original 21
slides is rewritten except Future Scope's body, which was empty.

Usage:
    python build_deck.py in.pptx out.pptx
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

EMU = 914400

# ============================================================
# DESIGN TOKENS  (measured off slides 11 and 15 of the original)
# ============================================================

TITLE_BLUE = "004AAD"      # every content-slide title
BODY_INK = "1E2936"        # every body paragraph
BRAND_RED = "A42530"       # the Canva template's own accent

# The deck already uses these three to mean face / voice / text, so they
# are kept pointing at the same ideas.
BLUE = "1C9CC0"
AMBER = "B06E00"
PURPLE = "7A5EA6"

# Pale card fills: a 20% luminance tint of a theme accent, which is how
# the existing cards are filled.
FILL_FOR = {BLUE: "accent1", AMBER: "accent2", PURPLE: "accent4",
            BRAND_RED: "accent2"}

HEAD_FONT = "Canva Sans Bold"
BODY_FONT = "Canva Sans"

CONTENT_LEFT = 1.25
CONTENT_WIDTH = 17.50
COL_X = [1.25, 7.30, 13.35]     # three columns, 5.35 wide, 0.70 gutter
COL_W = 5.35
RULE_H = 0.055                  # the 50292 EMU accent rule above a card

# The decoration occupies the bottom 1.03 inch of the 11.25 inch canvas.
SAFE_BOTTOM = 10.10


def emu(inches: float) -> int:
    return int(round(inches * EMU))


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ============================================================
# SHAPE BUILDERS
# ============================================================
#
# A paragraph is {runs, align, space_before}; a run is
# {text, size, bold, color}. The `text`/`size`/`bold` shorthand on a
# paragraph builds a single run, because most paragraphs have one.
# ============================================================


def _runs_of(para: dict) -> list[dict]:
    if "runs" in para:
        return para["runs"]
    return [{
        "text": para["text"],
        "size": para["size"],
        "bold": para.get("bold", False),
        "color": para.get("color", BODY_INK),
    }]


def _text_shape(
    shape_id: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    paragraphs: list[dict],
    fill: str | None = None,
) -> str:
    fill_xml = (
        f'<a:solidFill><a:schemeClr val="{fill}">'
        f'<a:lumMod val="20000"/><a:lumOff val="80000"/>'
        f"</a:schemeClr></a:solidFill>"
        if fill
        else "<a:noFill/>"
    )

    body = []
    for para in paragraphs:
        align = para.get("align", "l")
        before = para.get("space_before", 0)

        rendered = ""
        last = None
        for run in _runs_of(para):
            if not run["text"]:
                continue
            size = int(run["size"] * 100)
            bold = "1" if run.get("bold") else "0"
            font = HEAD_FONT if run.get("bold") else BODY_FONT
            color = run.get("color", BODY_INK)
            rendered += (
                f'<a:r><a:rPr lang="en-US" sz="{size}" b="{bold}" dirty="0">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{font}"/></a:rPr>'
                f"<a:t>{esc(run['text'])}</a:t></a:r>"
            )
            last = (size, bold, font, color)

        if last is None:
            first = _runs_of(para)[0]
            last = (int(first["size"] * 100), "0", BODY_FONT, BODY_INK)

        size, bold, font, color = last
        body.append(
            f'<a:p><a:pPr algn="{align}">'
            f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef></a:pPr>'
            f"{rendered}"
            f'<a:endParaRPr lang="en-US" sz="{size}" b="{bold}" dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{font}"/></a:endParaRPr></a:p>'
        )

    return (
        f"<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{name}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}</p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" anchor="t"><a:spAutoFit/></a:bodyPr>'
        f"<a:lstStyle/>{''.join(body)}</p:txBody></p:sp>"
    )


def _rule(shape_id: int, name: str, x: float, y: float, w: float,
          color: str) -> str:
    """The thin accent rule the deck draws above each card."""

    return (
        f"<p:sp><p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/>'
        f"</p:nvSpPr>"
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(RULE_H)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f"<a:ln><a:noFill/></a:ln><a:effectLst/></p:spPr>"
        f'<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/>'
        f'<a:p><a:pPr algn="ctr"/><a:endParaRPr/></a:p></p:txBody></p:sp>'
    )


# ============================================================
# TEXT FITTING
# ============================================================
#
# LibreOffice cannot render in the build container, so box heights are
# checked arithmetically instead of visually. Canva Sans is a humanist
# sans of roughly Open Sans' metrics: ~0.50 em average advance for mixed
# case prose, 1.22 em line height. Both are deliberate over-estimates,
# so a box that fits here fits in PowerPoint.
# ============================================================

CHAR_EM = 0.50
LINE_EM = 1.22
PAD = 0.18          # bodyPr default inset, both sides, in inches


def text_height(paragraphs: list[dict], width: float) -> float:
    total = 0.0
    for para in paragraphs:
        runs = _runs_of(para)
        chars = sum(len(r["text"]) for r in runs)
        size = max(r["size"] for r in runs)
        size_in = size / 72.0
        usable = width - 2 * PAD
        per_line = max(1, int(usable / (size_in * CHAR_EM)))
        lines = max(1, -(-chars // per_line))
        total += lines * size_in * LINE_EM
        total += para.get("space_before", 0) / 100.0 / 72.0
    return total + 0.12


def labelled(label: str, text: str, accent: str, size: float = 15,
             space_before: int = 500) -> dict:
    """'Label — body text', the label bold in the card's accent colour.

    Short labelled lines read from the back of a room; a paragraph of
    prose in a card does not. Every new card on these slides uses them.
    """

    return {
        "runs": [
            {"text": f"{label}  ", "size": size, "bold": True,
             "color": accent},
            {"text": text, "size": size, "color": BODY_INK},
        ],
        "space_before": space_before,
    }


# ============================================================
# SLIDE ASSEMBLY
# ============================================================


class SlideBuilder:
    def __init__(self, scaffold: str, title: str, subtitle: str | None = None):
        self.scaffold = scaffold
        self.next_id = 100
        self.body = ""
        self.title = title
        self.subtitle = subtitle
        self.max_y = 0.0

    def ids(self, count: int = 1) -> list[int]:
        got = list(range(self.next_id, self.next_id + count))
        self.next_id += count
        return got

    def add(self, xml: str, bottom: float) -> None:
        self.body += xml
        self.max_y = max(self.max_y, bottom)

    def text(self, x: float, y: float, w: float, paragraphs: list[dict],
             name: str = "Text") -> float:
        height = text_height(paragraphs, w)
        (shape_id,) = self.ids()
        self.add(_text_shape(shape_id, f"{name} {shape_id}", x, y, w, height,
                             paragraphs), y + height)
        return y + height

    def cards(self, y: float, items: list[dict], heading_size: float = 24)  -> float:
        """Three columns. Each item is {accent, heading, lines}.

        Rendered twice: once to find the tallest, once at a common height
        so the three cards bottom out together.
        """

        def paragraphs_for(item: dict) -> list[dict]:
            return [
                {"text": item["heading"], "size": heading_size, "bold": True,
                 "color": item["accent"], "align": "ctr"},
            ] + item["lines"]

        tallest = max(text_height(paragraphs_for(i), COL_W) for i in items)

        for column, item in enumerate(items):
            rule_id, card_id = self.ids(2)
            self.add(_rule(rule_id, f"Rule {rule_id}", COL_X[column],
                           y - 0.20, COL_W, item["accent"]), y)
            self.add(_text_shape(card_id, f"Card {card_id}", COL_X[column], y,
                                 COL_W, tallest, paragraphs_for(item),
                                 fill=FILL_FOR[item["accent"]]), y + tallest)

        return y + tallest

    def row(self, y: float, columns: list[tuple[float, float, dict]]) -> float:
        """One row of a comparison grid: (x, width, paragraph)."""

        bottom = y
        for x, width, para in columns:
            bottom = max(bottom, self.text(x, y, width, [para], name="Cell"))
        return bottom

    def render(self) -> str:
        head = _text_shape(
            10, "TextBox Title", CONTENT_LEFT, 0.52, CONTENT_WIDTH, 0.85,
            [{"text": self.title, "size": 40, "bold": True,
              "color": TITLE_BLUE}],
        )

        sub = ""
        if self.subtitle:
            sub = _text_shape(
                11, "TextBox Subtitle", 1.28, 1.52, 17.40, 0.60,
                [{"text": self.subtitle, "size": 17, "color": BODY_INK}],
            )

        return self.scaffold.replace("<!--BODY-->", head + sub + self.body)

    def check(self, label: str) -> None:
        if self.max_y > SAFE_BOTTOM:
            raise SystemExit(
                f"{label}: content reaches y={self.max_y:.2f}in, past the "
                f"{SAFE_BOTTOM}in decoration line"
            )
        print(f"  {label}: lowest content at y={self.max_y:.2f}in")


def footer(slide: SlideBuilder, y: float, heading: str, text: str,
           color: str = BODY_INK) -> None:
    slide.text(
        CONTENT_LEFT, y, CONTENT_WIDTH,
        [
            {"text": heading, "size": 20, "bold": True, "color": color},
            {"text": text, "size": 16, "space_before": 500},
        ],
        name="Footer",
    )


# ============================================================
# CONTENT
# ============================================================
#
# Every figure below is quoted from a committed result file on
# multimodal-v7-audio and is named in the speaker notes:
#
#   audio-emotion-module/results/benchmark.json, multimodal.json
#   text-emotion-module/results/tables.md
#   vit-emotion-v2-deployment/results_face_eval.v4.unseen_test.json
#   vit-emotion-v2-deployment/results_live_prior.json
#   vit-emotion-v2-deployment/reliability.json, face_pipeline.py
#   training/README.md
# ============================================================


def build_tech_stack(scaffold: str) -> str:
    slide = SlideBuilder(
        scaffold,
        "Technology Stack",
        "Three models, one Python process, no framework on the client - "
        "every choice sized to run on a single machine.",
    )

    bottom = slide.cards(3.15, [
        {
            "accent": BLUE,
            "heading": "Train",
            "lines": [
                labelled("Framework", "PyTorch 2.x + HuggingFace Transformers",
                         BLUE, space_before=900),
                labelled("Face", "ViT-Base/16, fine-tuned on four merged "
                                 "face corpora", BLUE),
                labelled("Voice", "WavLM-base+, 23k clips, speaker-disjoint "
                                  "splits", BLUE),
                labelled("Text", "TinyBERT, fine-tuned on GoEmotions", BLUE),
                labelled("Metrics", "scikit-learn; librosa + soundfile "
                                    "condition audio", BLUE),
                labelled("Hardware", "one RTX 4060. No cluster, no cloud.",
                         BLUE),
            ],
        },
        {
            "accent": AMBER,
            "heading": "Serve",
            "lines": [
                labelled("API", "FastAPI + Uvicorn", AMBER, space_before=900),
                labelled("Transport", "ONE WebSocket carries frames, 16 kHz "
                                      "PCM and text", AMBER),
                labelled("Face", "ONNX Runtime, CUDA provider, CPU fallback",
                         AMBER),
                labelled("Voice / Text", "PyTorch; the 380 MB voice model "
                                         "loads only if asked", AMBER),
                labelled("Detection", "OpenCV YuNet + five-point eye-line "
                                      "alignment", AMBER),
                labelled("Fusion", "~40 lines of NumPy; constants live in "
                                   "reliability.json", AMBER),
            ],
        },
        {
            "accent": PURPLE,
            "heading": "Show",
            "lines": [
                labelled("Client", "one HTML file. No React, no build step.",
                         PURPLE, space_before=900),
                labelled("Video", "getUserMedia to canvas to Base64 JPEG",
                         PURPLE),
                labelled("Audio", "AudioWorklet, mono 16 kHz windows", PURPLE),
                labelled("Pacing", "next frame only when the server "
                                   "acknowledges the last", PURPLE),
                labelled("Output", "label, three opinions, three weights, "
                                   "conflict flag", PURPLE),
                labelled("Cost", "32 FPS face, ~16 ms a voice window, "
                                 "~3 ms a message", PURPLE),
            ],
        },
    ])

    footer(
        slide, bottom + 0.30,
        "Why it is deliberately boring",
        "A telemedicine deployment cannot assume a datacentre. Every piece "
        "here degrades to CPU on one machine, and the heaviest piece is "
        "opt-in - a session that never uses the microphone never loads it.",
    )

    slide.check("Technology Stack")
    return slide.render()


def build_encoders(scaffold: str) -> str:
    slide = SlideBuilder(
        scaffold,
        "Meet the Three Encoders",
        "What each model actually is, and the one property that makes it "
        "right for its signal.",
    )

    bottom = slide.cards(3.15, [
        {
            "accent": BLUE,
            "heading": "ViT-Base/16  ·  the face",
            "lines": [
                labelled("What it is", "A transformer that cuts the face into "
                                       "16x16 patches and lets every patch "
                                       "look at every other patch at once.",
                         BLUE, space_before=900),
                labelled("Why that fits", "Angry and sad differ in how the "
                                          "brow and the mouth move TOGETHER. "
                                          "Attention links two distant "
                                          "regions in one step; a small-kernel "
                                          "CNN needs depth to reach across "
                                          "the face.", BLUE),
                labelled("In our system", "86M parameters, fine-tuned on four "
                                          "face datasets, exported to ONNX. "
                                          "6.6 ms a frame.", BLUE),
            ],
        },
        {
            "accent": AMBER,
            "heading": "WavLM-base+  ·  the voice",
            "lines": [
                labelled("What it is", "A speech model pre-trained by "
                                       "listening to ~94,000 hours of "
                                       "unlabelled speech and learning to "
                                       "fill in masked-out audio. It saw no "
                                       "emotion labels at that stage.",
                         AMBER, space_before=900),
                labelled("Why that fits", "That pre-training teaches it what "
                                          "voices DO - pitch, timing, strain. "
                                          "Our 4,179 labelled clips could "
                                          "never teach that from scratch.",
                         AMBER),
                labelled("In our system", "95M parameters. Scores a 4-second "
                                          "window about once a second, in "
                                          "16 ms.", AMBER),
            ],
        },
        {
            "accent": PURPLE,
            "heading": "TinyBERT 4L  ·  the words",
            "lines": [
                labelled("What it is", "A BERT shrunk by distillation: a full "
                                       "size model teaches a four-layer "
                                       "student to copy its answers. 14.4M "
                                       "parameters, about an eighth of "
                                       "BERT-base.", PURPLE,
                         space_before=900),
                labelled("Why that fits", "Text is the lightest signal and "
                                          "must not take the budget. At 2.8 ms "
                                          "a message it runs on CPU beside "
                                          "two other models.", PURPLE),
                labelled("In our system", "The GoEmotions checkpoint, because "
                                          "it predicts our seven classes "
                                          "directly.", PURPLE),
            ],
        },
    ])

    footer(
        slide, bottom + 0.30,
        "The one rule they all obey",
        "Each outputs a probability over the SAME seven emotions in the same "
        "order, so fusing them is arithmetic rather than translation. That "
        "constraint - not accuracy - is what ruled several otherwise better "
        "models out.",
    )

    slide.check("Meet the Three Encoders")
    return slide.render()


def build_alternatives(scaffold: str) -> str:
    slide = SlideBuilder(
        scaffold,
        "Why These, Not the Alternatives",
        "Each choice was measured against a named competitor on the same "
        "data. Nothing here came off a leaderboard.",
    )

    columns = [
        (1.25, 2.30),    # what
        (3.75, 3.30),    # chosen
        (7.35, 5.10),    # measured against
        (12.75, 6.00),   # result
    ]

    header = [
        (columns[0][0], columns[0][1],
         {"text": "WHAT", "size": 14, "bold": True, "color": TITLE_BLUE}),
        (columns[1][0], columns[1][1],
         {"text": "WE CHOSE", "size": 14, "bold": True, "color": TITLE_BLUE}),
        (columns[2][0], columns[2][1],
         {"text": "MEASURED AGAINST", "size": 14, "bold": True,
          "color": TITLE_BLUE}),
        (columns[3][0], columns[3][1],
         {"text": "WHAT THE MEASUREMENT SAID", "size": 14, "bold": True,
          "color": TITLE_BLUE}),
    ]

    rows = [
        ("Face", "ViT-Base/16 (v4)",
         "Our own v5 (balanced sampling), v6 (label cleaning) and a "
         "v3+v4 ensemble",
         "All three scored WORSE on unseen faces. v6 gained 3.6 points of "
         "validation accuracy and lost 4.3 on unseen faces - so the "
         "constraint is data diversity, not the model."),
        ("Voice", "WavLM-base+",
         "This project's own MFCC-CNN, retrained unchanged on identical "
         "speaker-disjoint splits",
         "Macro-F1 0.322 to 0.499. A 17.7-point gap with the data held "
         "constant, so it is attributable to the model."),
        ("Text", "TinyBERT 4L-312D",
         "DistilBERT (67M) and MobileBERT (24.6M), same split",
         "88.6 macro-F1 against 89.3 and 89.4. We give up 0.8 points and get "
         "2.8 ms instead of 10.6 and 16.9."),
        ("Text head", "GoEmotions checkpoint",
         "The higher-scoring 6-class MTEB checkpoint",
         "MTEB cannot express 'neutral' or 'disgust'. Neutral is the most "
         "common state in a consultation, so the better score was unusable."),
        ("Fusion", "Logarithmic opinion pool",
         "A learned fusion head",
         "No trimodal corpus exists in this label space. A learned head "
         "would be fitted on simulated pairs and could not be audited."),
    ]

    y = 2.70
    y = slide.row(y, header) + 0.12

    for what, chosen, against, result in rows:
        y = slide.row(y + 0.14, [
            (columns[0][0], columns[0][1],
             {"text": what, "size": 16, "bold": True, "color": BODY_INK}),
            (columns[1][0], columns[1][1],
             {"text": chosen, "size": 15, "bold": True, "color": BRAND_RED}),
            (columns[2][0], columns[2][1],
             {"text": against, "size": 15}),
            (columns[3][0], columns[3][1],
             {"text": result, "size": 15}),
        ])

    footer(
        slide, y + 0.28,
        "How this relates to slides 6 and 7",
        "Those slides proposed Combination A - MentalBERT, WavLM and DINOv2 - "
        "from a literature comparison. One finding survived into the built "
        "system: WavLM. MentalBERT and DINOv2 are not in the codebase; "
        "neither predicts our seven classes directly, and both exceed the CPU "
        "budget. Treat the F1 figures on slide 7 as that earlier study's, "
        "not this system's.",
        color=BRAND_RED,
    )

    slide.check("Why These, Not the Alternatives")
    return slide.render()


def build_stability(scaffold: str) -> str:
    slide = SlideBuilder(
        scaffold,
        "The Honest Problem: The Label Will Not Sit Still",
        "Hold a steady expression and the displayed emotion still changes "
        "several times a second. Three measured causes.",
    )

    bottom = slide.cards(3.15, [
        {
            "accent": BRAND_RED,
            "heading": "The top-1 is nearly a tie",
            "lines": [
                labelled("The number", "52.3% over seven classes on unseen "
                                       "faces.", BRAND_RED, space_before=900),
                labelled("What it means", "The leading class and the "
                                          "runner-up usually sit within a few "
                                          "points. Any change in crop, light "
                                          "or pose re-orders them - so the "
                                          "answer moves while the face does "
                                          "not.", BRAND_RED),
                labelled("Worst pairs", "neutral against sad, angry against "
                                        "disgust. Disgust holds only 0.20 F1 "
                                        "cross-dataset.", BRAND_RED),
            ],
        },
        {
            "accent": BRAND_RED,
            "heading": "The session prior is wrong",
            "lines": [
                labelled("The number", "Neutral is shown on 27.5% of frames; "
                                       "a real session is ~55% neutral.",
                         BRAND_RED, space_before=900),
                labelled("What it means", "The model is calibrated on a "
                                          "balanced set and carries that "
                                          "prior. Sad fires at 2.2x and "
                                          "surprise at 2.1x their true rate - "
                                          "most spurious 'surprise' flashes "
                                          "are a neutral face leaking.",
                         BRAND_RED),
                labelled("The fix exists", "Correcting the prior lifts "
                                           "accuracy 56.7% to 72.3%. It ships "
                                           "switched off.", BRAND_RED),
            ],
        },
        {
            "accent": BRAND_RED,
            "heading": "Only the face leg is smoothed",
            "lines": [
                labelled("The number", "EMA time constant 0.6 s, hysteresis "
                                       "margin 0.05 - inside the face "
                                       "pipeline only.", BRAND_RED,
                         space_before=900),
                labelled("What it means", "The fusion engine is stateless by "
                                          "design, so the number the clinician "
                                          "reads has no temporal filter at "
                                          "all.", BRAND_RED),
                labelled("And worse", "Its weights move every frame as audio "
                                      "decays (8 s half-life) and text decays "
                                      "(30 s). The reading can change with no "
                                      "new evidence.", BRAND_RED),
            ],
        },
    ])

    footer(
        slide, bottom + 0.30,
        "And nothing in this deck measures it",
        "Every figure we have shown is per-frame accuracy on still images. "
        "Not one metric describes how often the displayed label changes - "
        "which is the first thing a clinician would notice. That gap is item "
        "one in the future scope.",
        color=BRAND_RED,
    )

    slide.check("Stability")
    return slide.render()


def build_future_scope(scaffold: str) -> str:
    slide = SlideBuilder(
        scaffold,
        "Future Scope",
        "Ordered, not listed: the instability has to be measured and closed "
        "before any clinical claim is worth making.",
    )

    slide.text(
        CONTENT_LEFT, 2.12, CONTENT_WIDTH,
        [{"text": "NEXT  —  make the live output trustworthy",
          "size": 17, "bold": True, "color": TITLE_BLUE}],
        name="RowLabel",
    )

    row_one = slide.cards(2.90, [
        {
            "accent": BLUE,
            "heading": "1 · Measure stability",
            "lines": [
                labelled("Do", "Report switches per minute, mean dwell time "
                               "and sequence entropy beside accuracy.", BLUE,
                         space_before=800),
                labelled("Why", "52% accurate and steady is usable; 52% and "
                                "flickering is not. Today's metrics cannot "
                                "tell them apart.", BLUE),
            ],
        },
        {
            "accent": AMBER,
            "heading": "2 · Smooth the decision",
            "lines": [
                labelled("Do", "Move the EMA and hysteresis after the pool, "
                               "then replace both with a seven-state HMM "
                               "decoded by Viterbi.", AMBER, space_before=800),
                labelled("Why", "Changing the reported emotion should have to "
                                "be paid for by evidence. Also: turn the "
                                "prior correction on.", AMBER),
            ],
        },
        {
            "accent": PURPLE,
            "heading": "3 · Report episodes",
            "lines": [
                labelled("Do", "Aggregate to intervals - '2 min 40 s "
                               "predominantly sad, two conflicts' - with "
                               "per-frame output on request.", PURPLE,
                         space_before=800),
                labelled("Why", "A clinician does not need 32 labels a "
                                "second, and this gives the system somewhere "
                                "to say 'unknown'.", PURPLE),
            ],
        },
    ], heading_size=21)

    slide.text(
        CONTENT_LEFT, row_one + 0.22, CONTENT_WIDTH,
        [{"text": "THEN  —  what a clinical claim would require",
          "size": 17, "bold": True, "color": TITLE_BLUE}],
        name="RowLabel",
    )

    slide.cards(row_one + 1.00, [
        {
            "accent": BLUE,
            "heading": "4 · Escape the data ceiling",
            "lines": [
                labelled("Do", "Full AffectNet (~287k against our 27,823) and "
                               "a real trimodal corpus - IEMOCAP or "
                               "CMU-MOSEI.", BLUE, space_before=800),
                labelled("Why", "Three attempts to beat v4 all failed, which "
                                "says the constraint is face diversity, not "
                                "capacity.", BLUE),
            ],
        },
        {
            "accent": AMBER,
            "heading": "5 · Validate clinically",
            "lines": [
                labelled("Do", "Agreement against clinician ratings on real "
                               "consultations, under ethics approval.", AMBER,
                         space_before=800),
                labelled("Why", "Every number here is acted corpora or TV "
                                "dialogue. Report accuracy per skin tone, age "
                                "and gender.", AMBER),
            ],
        },
        {
            "accent": PURPLE,
            "heading": "6 · Privacy by construction",
            "lines": [
                labelled("Do", "Run the face leg in the browser with ONNX "
                               "Runtime Web; send only the seven "
                               "probabilities.", PURPLE, space_before=800),
                labelled("Why", "Frames currently cross the wire as Base64 "
                                "JPEG. Federated training then reaches data "
                                "that cannot be centralised.", PURPLE),
            ],
        },
    ], heading_size=21)

    slide.check("Future Scope")
    return slide.render()


# ============================================================
# PACKAGE SURGERY
# ============================================================

CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
)

# Each new slide, and the ORIGINAL slide number it is shown after.
INSERT_AFTER = [
    (10, build_tech_stack),
    (11, build_encoders),
    (11, build_alternatives),
    (18, build_stability),
]


def make_scaffold(slide19_xml: str) -> str:
    """Strip Future Scope down to its decoration, leaving a BODY marker.

    The four freeform groups are the Canva template's furniture - the red
    bar along the bottom, two logo images, the corner mark. They are
    identical on every content slide, so they become the scaffold.
    """

    start = slide19_xml.index("<p:grpSp>")
    end = slide19_xml.rindex("</p:grpSp>") + len("</p:grpSp>")
    decoration = slide19_xml[start:end]

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships"'
        ' xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>"
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        f"{decoration}<!--BODY--></p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


def main() -> None:
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])

    with zipfile.ZipFile(source) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}

    slide19 = parts["ppt/slides/slide19.xml"].decode("utf-8")
    scaffold = make_scaffold(slide19)
    slide19_rels = parts["ppt/slides/_rels/slide19.xml.rels"]

    print("Building new slides:")
    built = [(after, builder(scaffold)) for after, builder in INSERT_AFTER]

    parts["ppt/slides/slide19.xml"] = build_future_scope(scaffold).encode("utf-8")

    numbers = [
        int(m.group(1))
        for name in parts
        if (m := re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name))
    ]
    next_number = max(numbers) + 1

    added = []
    for after, xml in built:
        parts[f"ppt/slides/slide{next_number}.xml"] = xml.encode("utf-8")
        parts[f"ppt/slides/_rels/slide{next_number}.xml.rels"] = slide19_rels
        added.append((after, next_number))
        next_number += 1

    # --- content types -------------------------------------------------
    types = parts["[Content_Types].xml"].decode("utf-8")
    types = types.replace("</Types>", "".join(
        f'<Override PartName="/ppt/slides/slide{number}.xml" '
        f'ContentType="{CONTENT_TYPE}"/>' for _, number in added
    ) + "</Types>")
    parts["[Content_Types].xml"] = types.encode("utf-8")

    # --- presentation rels --------------------------------------------
    rels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    next_rid = max(int(m) for m in re.findall(r'Id="rId(\d+)"', rels)) + 1

    rid_for = {}
    new_rels = ""
    for after, number in added:
        rid_for[number] = f"rId{next_rid}"
        new_rels += (
            f'<Relationship Id="rId{next_rid}" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{number}.xml"/>'
        )
        next_rid += 1

    parts["ppt/_rels/presentation.xml.rels"] = rels.replace(
        "</Relationships>", new_rels + "</Relationships>"
    ).encode("utf-8")

    # --- slide order ---------------------------------------------------
    presentation = parts["ppt/presentation.xml"].decode("utf-8")
    id_list = re.search(r"<p:sldIdLst>.*?</p:sldIdLst>", presentation, re.S)
    entries = re.findall(r"<p:sldId [^/]*/>", id_list.group(0))

    if len(entries) != 21:
        raise SystemExit(f"expected 21 slides, found {len(entries)}")

    next_sld_id = max(
        max(int(m) for m in re.findall(r'id="(\d+)"', id_list.group(0))), 255
    ) + 1

    inserts: dict[int, list[str]] = {}
    for after, number in added:
        inserts.setdefault(after, []).append(
            f'<p:sldId id="{next_sld_id}" r:id="{rid_for[number]}"/>'
        )
        next_sld_id += 1

    ordered = []
    for position, entry in enumerate(entries, start=1):
        ordered.append(entry)
        ordered.extend(inserts.get(position, []))

    parts["ppt/presentation.xml"] = presentation.replace(
        id_list.group(0), f"<p:sldIdLst>{''.join(ordered)}</p:sldIdLst>"
    ).encode("utf-8")

    # --- write ---------------------------------------------------------
    if target.exists():
        target.unlink()

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)

    print(f"\nWrote {target} with {len(ordered)} slides")


if __name__ == "__main__":
    main()
