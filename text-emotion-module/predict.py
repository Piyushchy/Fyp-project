# ============================================================
# TEXT EMOTION MODULE - INFERENCE
# ============================================================
#
# The deployment-facing entry point. Mirrors the role that
# realtime_inference_onnx.py plays for the ViT V2 visual module,
# but for the text modality.
#
#   python predict.py "i cant believe you remembered my birthday"
#   python predict.py --model distilbert "everything is falling apart"
#   python predict.py --examples
#   python predict.py --interactive
#
# Output shows the MTEB emotion label, the confidence, and the
# shared label the fusion module consumes.
# ============================================================

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    ID2LABEL,
    MODELS,
    TEXT_TO_SHARED_LABEL,
    checkpoint_path,
)
from modeling import load_model


# ============================================================
# EXAMPLE SENTENCES
# ============================================================
#
# Two per emotion, held out from the dataset - these are written
# for the demo, not sampled from the test split, so the printed
# output is not an evaluation result. Use benchmark.py for that.
# ============================================================

EXAMPLES = [
    ("i feel so empty since she stopped calling me back", "sadness"),
    ("i keep replaying the goodbye and it still hurts", "sadness"),
    ("i cant believe you remembered my birthday this is wonderful", "joy"),
    ("i am so proud of how far we have come together", "joy"),
    ("i adore the way she laughs at her own jokes", "love"),
    ("i feel such tenderness whenever he holds my hand", "love"),
    ("i am furious that they lied to me again", "anger"),
    ("it makes my blood boil when people talk over me", "anger"),
    ("i am terrified of what the results will say tomorrow", "fear"),
    ("i feel a knot in my stomach every time the phone rings", "fear"),
    ("i did not expect a single one of them to show up", "surprise"),
    ("i am stunned that this actually worked on the first try", "surprise"),
]


# ============================================================
# CLASSIFIER
# ============================================================

class TextEmotionClassifier:
    """Wraps one fine-tuned checkpoint for single-sentence inference."""

    def __init__(self, model_key="tinybert", device="cpu"):

        directory = checkpoint_path(model_key)

        if not directory.exists():
            raise FileNotFoundError(
                f"No checkpoint at {directory}.\n"
                f"Train it first:  python train.py --model {model_key}"
            )

        self.model_key = model_key
        self.display_name = MODELS[model_key]["display_name"]
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(directory)

        self.model = load_model(model_key, directory).to(device)

        # eval() also switches MobileBERT's pooled BatchNorm to its
        # running statistics, so a single sentence classifies the
        # same way it would inside a batch.
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        """Return the predicted emotion, confidence, timing and full distribution."""

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MODELS[self.model_key]["max_length"],
            return_tensors="pt",
        ).to(self.device)

        started = time.perf_counter()

        logits = self.model(**encoded).logits

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        probabilities = F.softmax(logits, dim=-1)[0]

        index = int(probabilities.argmax())

        label = ID2LABEL[index]

        return {
            "text": text,
            "label": label,
            "confidence": float(probabilities[index]),
            "shared_label": TEXT_TO_SHARED_LABEL[label],
            "latency_ms": elapsed_ms,
            "distribution": {
                ID2LABEL[i]: float(probabilities[i])
                for i in range(len(ID2LABEL))
            },
        }


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def print_prediction(prediction, show_distribution=False):

    print(f'  "{prediction["text"]}"')

    print(f"  -> {prediction['label']} "
          f"({prediction['confidence']*100:.1f}%)  "
          f"[fusion: {prediction['shared_label']}]  "
          f"{prediction['latency_ms']:.1f} ms")

    if show_distribution:

        ordered = sorted(
            prediction["distribution"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for name, probability in ordered:
            bar = "#" * int(round(probability * 30))
            print(f"     {name:<9} {probability*100:5.1f}%  {bar}")

    print()


# ============================================================
# MODES
# ============================================================

def run_examples(classifier, show_distribution):

    print(f"Model: {classifier.display_name}\n")

    correct = 0

    for text, expected in EXAMPLES:

        prediction = classifier.predict(text)

        matched = prediction["label"] == expected

        correct += matched

        print(f"[{'ok ' if matched else 'MISS'}] expected {expected}")

        print_prediction(prediction, show_distribution)

    print(f"{correct}/{len(EXAMPLES)} demo sentences matched the intended emotion.")


def run_interactive(classifier, show_distribution):

    print(f"Model: {classifier.display_name}")
    print("Type a sentence and press Enter. Blank line or Ctrl-C to quit.\n")

    while True:

        try:
            text = input("text > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            break

        print_prediction(classifier.predict(text), show_distribution)

    print("Closed.")


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "text",
        nargs="*",
        help="Sentence to classify.",
    )

    parser.add_argument(
        "--model",
        default="tinybert",
        choices=list(MODELS),
        help="Which fine-tuned checkpoint to load (default: tinybert).",
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run the built-in demo sentences.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read sentences from stdin in a loop.",
    )

    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Show the probability of every emotion, not just the top one.",
    )

    parser.add_argument("--device", default="cpu")

    arguments = parser.parse_args()

    classifier = TextEmotionClassifier(arguments.model, arguments.device)

    if arguments.examples:
        run_examples(classifier, arguments.distribution)

    elif arguments.interactive:
        run_interactive(classifier, arguments.distribution)

    elif arguments.text:
        print()
        print_prediction(
            classifier.predict(" ".join(arguments.text)),
            arguments.distribution,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
