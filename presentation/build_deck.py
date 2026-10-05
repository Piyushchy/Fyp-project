"""
Rebuild Sept_eval_1_multimodal.pptx with the slides the September review
left open:

  * a Technology Stack slide                      (new, after User Flow)
  * a Why These Models slide                      (new, after Why Three Modalities)
  * a Label Stability slide                       (new, after Inside the Interface)
  * Future Scope                                  (was an empty title)

and speaker notes on every slide.

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

import copy
import re
import shutil
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

# Column accents. The deck already uses these three to mean
# face / voice / text, so they are kept pointing at the same ideas.
BLUE = "1C9CC0"
AMBER = "B06E00"
PURPLE = "7A5EA6"

# Pale card fills: a 20% luminance tint of a theme accent, which is how
# the existing cards are filled.
FILL_FOR = {BLUE: "accent1", AMBER: "accent2", PURPLE: "accent4", BRAND_RED: "accent2"}

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
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================
# SHAPE BUILDERS
# ============================================================


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
    """One text box. `paragraphs` is a list of
    {text, size, bold, color, align, space_before}."""

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
        size = int(para["size"] * 100)
        bold = "1" if para.get("bold") else "0"
        font = HEAD_FONT if para.get("bold") else BODY_FONT
        color = para.get("color", BODY_INK)

        run = ""
        if para["text"]:
            run = (
                f'<a:r><a:rPr lang="en-US" sz="{size}" b="{bold}" dirty="0">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{font}"/></a:rPr>'
                f"<a:t>{esc(para['text'])}</a:t></a:r>"
            )

        body.append(
            f'<a:p><a:pPr algn="{align}">'
            f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef></a:pPr>'
            f"{run}"
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


def _rule(shape_id: int, name: str, x: float, y: float, w: float, color: str) -> str:
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
        f"<p:txBody><a:bodyPr rtlCol=\"0\" anchor=\"ctr\"/><a:lstStyle/>"
        f'<a:p><a:pPr algn="ctr"/><a:endParaRPr/></a:p></p:txBody></p:sp>'
    )


# ============================================================
# TEXT FITTING
# ============================================================
#
# LibreOffice cannot render in the build container, so card heights are
# checked arithmetically instead of visually. Canva Sans is a humanist
# sans of roughly Open Sans' metrics: ~0.50 em average advance for mixed
# case prose, 1.22 em line height. Both are deliberate over-estimates,
# so a card that fits here fits in PowerPoint.
# ============================================================

CHAR_EM = 0.50
LINE_EM = 1.22
PAD = 0.18          # bodyPr default inset, both sides, in inches


def text_height(paragraphs: list[dict], width: float) -> float:
    total = 0.0
    for para in paragraphs:
        size_in = para["size"] / 72.0
        usable = width - 2 * PAD
        per_line = max(1, int(usable / (size_in * CHAR_EM)))
        lines = max(1, -(-len(para["text"]) // per_line))
        total += lines * size_in * LINE_EM
        total += para.get("space_before", 0) / 100.0 / 72.0
    return total + 0.10


def card(
    ids: list[int],
    column: int,
    y: float,
    height: float,
    accent: str,
    heading: str,
    body: str,
    heading_size: float = 26,
    body_size: float = 16,
) -> tuple[str, float]:
    """Accent rule + filled card, returned with the height it needs."""

    x = COL_X[column]
    paragraphs = [
        {"text": heading, "size": heading_size, "bold": True,
         "color": accent, "align": "ctr"},
        {"text": body, "size": body_size, "color": BODY_INK,
         "space_before": 900},
    ]

    needed = text_height(paragraphs, COL_W)
    used = max(height, needed)

    xml = _rule(ids[0], f"Rule {ids[0]}", x, y - 0.20, COL_W, accent)
    xml += _text_shape(
        ids[1], f"Card {ids[1]}", x, y, COL_W, used, paragraphs,
        fill=FILL_FOR[accent],
    )
    return xml, used


# ============================================================
# SLIDE ASSEMBLY
# ============================================================


class SlideBuilder:
    """Builds one slide on the decoration cloned from the template slide."""

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

    def text(
        self,
        x: float,
        y: float,
        w: float,
        paragraphs: list[dict],
        name: str = "Text",
    ) -> None:
        height = text_height(paragraphs, w)
        (shape_id,) = self.ids()
        self.add(
            _text_shape(shape_id, f"{name} {shape_id}", x, y, w, height,
                        paragraphs),
            y + height,
        )

    def cards(self, y: float, height: float, items: list[tuple[str, str, str]],
              heading_size: float = 26, body_size: float = 16) -> float:
        """Three columns of (accent, heading, body). Returns the row bottom."""

        tallest = height
        pieces = []
        for column, (accent, heading, body) in enumerate(items):
            xml, used = card(self.ids(2), column, y, height, accent,
                             heading, body, heading_size, body_size)
            pieces.append(xml)
            tallest = max(tallest, used)

        # Re-emit at a common height so the three cards align.
        self.next_id -= 2 * len(items)
        for column, (accent, heading, body) in enumerate(items):
            xml, _ = card(self.ids(2), column, y, tallest, accent,
                          heading, body, heading_size, body_size)
            self.add(xml, y + tallest)

        return y + tallest

    def render(self) -> str:
        # 0.85in is what one 40pt line actually occupies. The original
        # slides declare 1.35 and rely on spAutoFit to shrink it on open;
        # declaring the real height keeps the static geometry honest for
        # the overlap check, and renders identically.
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

        return self.scaffold.replace(
            "<!--BODY-->", head + sub + self.body
        )

    def check(self, label: str) -> None:
        if self.max_y > SAFE_BOTTOM:
            raise SystemExit(
                f"{label}: content reaches y={self.max_y:.2f}in, past the "
                f"{SAFE_BOTTOM}in decoration line"
            )
        print(f"  {label}: lowest content at y={self.max_y:.2f}in")


# ============================================================
# CONTENT
# ============================================================
#
# Every figure below is quoted from a committed result file on
# multimodal-v7-audio and is named in the speaker notes:
#
#   audio-emotion-module/results/benchmark.json
#   audio-emotion-module/results/multimodal.json
#   text-emotion-module/results/benchmark.json, tables.md
#   vit-emotion-v2-deployment/results_face_eval.v4.unseen_test.json
#   vit-emotion-v2-deployment/results_live_prior.json
#   vit-emotion-v2-deployment/reliability.json
# ============================================================


def build_tech_stack(scaffold: str) -> tuple[str, str]:
    slide = SlideBuilder(
        scaffold,
        "Technology Stack",
        "Three models, one process, no framework on the client - every "
        "choice sized to run on a single machine with one GPU.",
    )

    bottom = slide.cards(
        3.15,
        4.30,
        [
            (
                BLUE,
                "Train",
                "PyTorch 2.x with HuggingFace Transformers for all three "
                "encoders. The face ViT is fine-tuned from "
                "google/vit-base-patch16-224-in21k over four merged corpora; "
                "WavLM-base+ and TinyBERT are fine-tuned in the same "
                "framework. librosa and soundfile condition audio, "
                "scikit-learn computes every metric, and one RTX 4060 did "
                "all of the training.",
            ),
            (
                AMBER,
                "Serve",
                "FastAPI and Uvicorn behind a single WebSocket that carries "
                "video frames, 16 kHz PCM and typed text on one connection. "
                "The face ViT runs through ONNX Runtime on the CUDA provider "
                "and falls back to CPU; WavLM and TinyBERT stay in PyTorch. "
                "OpenCV's YuNet detector supplies the five landmarks the "
                "aligned crop needs. The pool itself is NumPy.",
            ),
            (
                PURPLE,
                "Show",
                "The client is one HTML file - no framework, no build step. "
                "getUserMedia and a canvas produce Base64 JPEG frames paced "
                "on server acknowledgements; an AudioWorklet produces mono "
                "16 kHz windows. Every fusion constant lives in "
                "reliability.json as data, so a weight can be retuned and "
                "cited without touching code.",
            ),
        ],
    )

    slide.text(
        CONTENT_LEFT,
        bottom + 0.25,
        CONTENT_WIDTH,
        [
            {"text": "Why it is this boring", "size": 20, "bold": True,
             "color": BODY_INK},
            {"text": "A telemedicine deployment cannot assume a datacentre. "
                     "Every component here degrades to CPU on one machine, "
                     "and the heaviest piece - the 380 MB WavLM checkpoint - "
                     "is opt-in, so a session that never uses the microphone "
                     "never pays for it.",
             "size": 16, "space_before": 600},
        ],
        name="Footer",
    )

    slide.check("Technology Stack")
    return slide.render(), "Technology Stack"


def build_why_models(scaffold: str) -> tuple[str, str]:
    slide = SlideBuilder(
        scaffold,
        "Why These Models",
        "Each encoder was chosen against a measured alternative on the same "
        "data, not off a leaderboard.",
    )

    bottom = slide.cards(
        3.15,
        4.30,
        [
            (
                BLUE,
                "ViT-Base/16",
                "Angry and sad differ in how the brow and the mouth move "
                "together, and self-attention over 16x16 patches relates "
                "those two regions in one layer where a small-kernel CNN "
                "needs depth to reach across the face. Capacity is not the "
                "binding constraint: balanced sampling (v5), label cleaning "
                "(v6) and a two-checkpoint ensemble all measured WORSE than "
                "v4 on unseen faces. ImageNet-21k pretraining is doing more "
                "work than any further architecture change would.",
            ),
            (
                AMBER,
                "WavLM-base+",
                "Benchmarked against this project's own MFCC CNN on "
                "identical speaker-disjoint splits: macro-F1 0.322 to 0.499, "
                "a 17.7-point gap from the model alone. Self-supervised "
                "pretraining on roughly 94k hours of speech is knowledge "
                "4,179 clips cannot supply. A learned mixture over its 13 "
                "hidden states settles on layers 9-10, not the top - the "
                "masked-prediction head drifts away from paralinguistics, so "
                "the final layer alone would discard the emotion.",
            ),
            (
                PURPLE,
                "TinyBERT 4L-312D",
                "Benchmarked against DistilBERT (67M) and MobileBERT (24.6M) "
                "on the same split: 88.6 macro-F1 against 89.3 and 89.4, so "
                "0.8 points behind at 2.8 ms against 10.6 and 16.9, and 7.6x "
                "smaller than BERT-base. BERT-base and RoBERTa were excluded "
                "on the CPU budget, not on accuracy. The deployed checkpoint "
                "is the GoEmotions one because it predicts the shared seven "
                "directly; the 6-class head cannot say neutral or disgust.",
            ),
        ],
    )

    slide.text(
        CONTENT_LEFT,
        bottom + 0.25,
        CONTENT_WIDTH,
        [
            {"text": "And why the fusion is not a fourth model",
             "size": 20, "bold": True, "color": BODY_INK},
            {"text": "A learned fusion head needs trimodal training data in "
                     "this label space, and none exists - it would have to be "
                     "fitted on simulated pairs and could never be audited. "
                     "The logarithmic opinion pool has no parameters to fit, "
                     "survives a missing modality, and reports the weight it "
                     "gave each signal, which is what a clinician actually "
                     "has to check.",
             "size": 16, "space_before": 600},
        ],
        name="Footer",
    )

    slide.check("Why These Models")
    return slide.render(), "Why These Models"


def build_stability(scaffold: str) -> tuple[str, str]:
    slide = SlideBuilder(
        scaffold,
        "The Honest Problem: The Label Will Not Sit Still",
        "Hold a steady expression and the displayed emotion still changes "
        "several times a second. Three measured causes - none of them the "
        "viewer's imagination.",
    )

    bottom = slide.cards(
        3.15,
        4.30,
        [
            (
                BRAND_RED,
                "The top-1 is nearly a tie",
                "52.3% accuracy over seven classes on unseen faces means the "
                "leading and the runner-up class are usually within a few "
                "points of each other. Any change in the crop, the lighting "
                "or the head pose re-orders them, so the argmax moves while "
                "the face does not. neutral against sad, and angry against "
                "disgust, are the pairs that swap most - disgust holds only "
                "0.20 F1 cross-dataset.",
            ),
            (
                BRAND_RED,
                "The session prior is wrong",
                "A webcam consultation is roughly 55% neutral, but the model "
                "is calibrated on a balanced set. Measured under a realistic "
                "prior it emits neutral on 27.5% of frames, sad at 2.2x and "
                "surprise at 2.1x their true rate - so most spurious "
                "'surprise' flashes are a neutral face leaking. Correcting "
                "the prior lifts accuracy from 56.7% to 72.3%, and the "
                "correction ships switched off.",
            ),
            (
                BRAND_RED,
                "Only the face leg is smoothed",
                "The EMA (0.6 s time constant) and the 0.05 hysteresis "
                "margin sit inside the face pipeline. The fusion engine is "
                "stateless by design, so the number on screen has no "
                "temporal filter at all - and its weights move every frame "
                "as audio decays with an 8 s half-life and text with a 30 s "
                "one. The reading can change with no new evidence.",
            ),
        ],
    )

    slide.text(
        CONTENT_LEFT,
        bottom + 0.25,
        CONTENT_WIDTH,
        [
            {"text": "And nothing in this deck measures it",
             "size": 20, "bold": True, "color": BRAND_RED},
            {"text": "Every figure presented so far is per-frame accuracy on "
                     "still images. Not one metric describes how often the "
                     "displayed label changes, which is the thing a clinician "
                     "would notice first. That gap is the first item in the "
                     "future scope, because nothing here can be fixed while "
                     "nothing measures it.",
             "size": 16, "space_before": 600},
        ],
        name="Footer",
    )

    slide.check("Stability")
    return slide.render(), "Stability"


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
          "size": 17, "bold": True, "color": BODY_INK}],
        name="RowLabel",
    )

    row_one = slide.cards(
        2.90,
        2.95,
        [
            (
                BLUE,
                "1 · Measure stability",
                "Report switches per minute, mean dwell time per label and "
                "the entropy of the displayed sequence over held-out video, "
                "beside accuracy. A model that is 52% accurate and steady is "
                "clinically usable; one that is 52% accurate and flickering "
                "is not, and today's metrics cannot tell them apart.",
            ),
            (
                AMBER,
                "2 · Smooth the decision",
                "Move the EMA and hysteresis after the pool rather than "
                "inside the face leg, then replace both with an explicit "
                "dwell-time model - a seven-state HMM decoded with Viterbi - "
                "so changing the reported emotion has to be paid for. Enable "
                "the prior correction and report it as a serving rule.",
            ),
            (
                PURPLE,
                "3 · Report episodes, not frames",
                "A clinician does not need 32 labels a second. Aggregate to "
                "intervals - '2 min 40 s predominantly sad, two flagged "
                "conflicts' - and expose per-frame output only on request. "
                "This also lets the system abstain on a low-quality or "
                "conflicted stretch instead of guessing.",
            ),
        ],
        heading_size=21,
        body_size=15,
    )

    slide.text(
        CONTENT_LEFT, row_one + 0.22, CONTENT_WIDTH,
        [{"text": "THEN  —  what a clinical claim would require",
          "size": 17, "bold": True, "color": BODY_INK}],
        name="RowLabel",
    )

    row_two = slide.cards(
        row_one + 1.00,
        2.95,
        [
            (
                BLUE,
                "4 · Escape the data ceiling",
                "v5, v6 and the ensemble all failed, which says the binding "
                "constraint is face diversity, not capacity. The levers left "
                "are the full AffectNet (~287k against the 27,823 public "
                "subset) and a trimodal corpus such as IEMOCAP or CMU-MOSEI, "
                "so fusion can be fitted and tested on real pairs rather "
                "than simulated ones.",
            ),
            (
                AMBER,
                "5 · Validate clinically",
                "Every number here comes from acted corpora or TV dialogue. "
                "'Emotionally intelligent telemedicine' needs agreement "
                "against clinician ratings on real consultations, under "
                "ethics approval, with accuracy reported per skin tone, age "
                "and gender - the fairness risk a face model carries by "
                "default.",
            ),
            (
                PURPLE,
                "6 · Privacy by construction",
                "Frames currently cross the wire as Base64 JPEG. Moving the "
                "face leg into the browser with ONNX Runtime Web keeps pixels "
                "on the device and sends only a probability vector. "
                "Federated fine-tuning is then the route to clinical data "
                "that cannot legally be centralised.",
            ),
        ],
        heading_size=21,
        body_size=15,
    )

    slide.check("Future Scope")
    return slide.render()


# ============================================================
# SPEAKER NOTES
# ============================================================

NOTES = {}   # filled from notes.py, keyed by final slide number


# ============================================================
# PACKAGE SURGERY
# ============================================================

CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "presentationml.slide+xml"
)


def make_scaffold(slide19_xml: str) -> str:
    """Strip Future Scope down to its decoration, leaving a BODY marker.

    The four freeform groups are the Canva template's furniture: the red
    bar along the bottom, the two logo images, the corner mark. They are
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
        "<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/>"
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

    new_slides = {
        # (template-slide to insert AFTER, built xml)
        "tech": (10, build_tech_stack(scaffold)[0]),
        "models": (11, build_why_models(scaffold)[0]),
        "stability": (18, build_stability(scaffold)[0]),
    }

    parts["ppt/slides/slide19.xml"] = build_future_scope(scaffold).encode("utf-8")

    # --- add the three new slide parts ---------------------------------
    numbers = [
        int(m.group(1))
        for name in parts
        if (m := re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name))
    ]
    next_number = max(numbers) + 1

    added = {}
    for key, (after, xml) in new_slides.items():
        name = f"ppt/slides/slide{next_number}.xml"
        parts[name] = xml.encode("utf-8")
        parts[f"ppt/slides/_rels/slide{next_number}.xml.rels"] = slide19_rels
        added[key] = (after, next_number)
        next_number += 1

    # --- content types -------------------------------------------------
    types = parts["[Content_Types].xml"].decode("utf-8")
    additions = "".join(
        f'<Override PartName="/ppt/slides/slide{number}.xml" '
        f'ContentType="{CONTENT_TYPE}"/>'
        for _, number in added.values()
    )
    types = types.replace("</Types>", additions + "</Types>")
    parts["[Content_Types].xml"] = types.encode("utf-8")

    # --- presentation rels --------------------------------------------
    rels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    existing = [int(m) for m in re.findall(r'Id="rId(\d+)"', rels)]
    next_rid = max(existing) + 1

    rid_for = {}
    new_rels = ""
    for key, (_, number) in added.items():
        rid_for[key] = f"rId{next_rid}"
        new_rels += (
            f'<Relationship Id="rId{next_rid}" Type="http://schemas.'
            "openxmlformats.org/officeDocument/2006/relationships/slide\" "
            f'Target="slides/slide{number}.xml"/>'
        )
        next_rid += 1

    rels = rels.replace("</Relationships>", new_rels + "</Relationships>")
    parts["ppt/_rels/presentation.xml.rels"] = rels.encode("utf-8")

    # --- slide order ---------------------------------------------------
    presentation = parts["ppt/presentation.xml"].decode("utf-8")
    id_list = re.search(r"<p:sldIdLst>.*?</p:sldIdLst>", presentation, re.S)
    entries = re.findall(r"<p:sldId [^/]*/>", id_list.group(0))

    if len(entries) != 21:
        raise SystemExit(f"expected 21 slides, found {len(entries)}")

    used_ids = [int(m) for m in re.findall(r'id="(\d+)"', id_list.group(0))]
    next_sld_id = max(max(used_ids), 255) + 1

    inserts = {}
    for key, (after, _) in added.items():
        entry = (
            f'<p:sldId id="{next_sld_id}" r:id="{rid_for[key]}"/>'
        )
        inserts.setdefault(after, []).append(entry)
        next_sld_id += 1

    ordered = []
    for position, entry in enumerate(entries, start=1):
        ordered.append(entry)
        ordered.extend(inserts.get(position, []))

    parts["ppt/presentation.xml"] = presentation.replace(
        id_list.group(0),
        f"<p:sldIdLst>{''.join(ordered)}</p:sldIdLst>",
    ).encode("utf-8")

    # --- write ---------------------------------------------------------
    if target.exists():
        target.unlink()

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)

    print(f"\nWrote {target} with {len(ordered)} slides")
    for key, (after, number) in added.items():
        print(f"  {key}: slide{number}.xml, shown after original slide {after}")


if __name__ == "__main__":
    main()
