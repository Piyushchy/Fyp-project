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

from calibration import load_temperature
from config import (
    DEFAULT_TASK,
    MODELS,
    SHARED_LABELS,
    TASKS,
    TEXT_TO_SHARED_LABEL,
    get_task,
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
    """Wraps one fine-tuned checkpoint for single-sentence inference.

    Serves both tasks. The GoEmotions checkpoints already predict in
    the shared seven-class space, so `shared_distribution` is their
    output unchanged. The older six-class MTEB checkpoints are
    projected through TEXT_TO_SHARED_LABEL, which folds 'love' into
    'happy' and leaves 'neutral' and 'disgust' at zero - they have no
    way to express either. That gap is the reason the GoEmotions task
    exists, and it is why fusion defaults to it.
    """

    def __init__(self, model_key="tinybert", device="cpu", task=DEFAULT_TASK):

        self.task = get_task(task) if isinstance(task, str) else task

        directory = self.task.checkpoint_path(model_key)

        if not directory.exists():
            raise FileNotFoundError(
                f"No checkpoint at {directory}.\n"
                f"Train it first:  python train.py "
                f"--task {self.task.key} --model {model_key}"
            )

        self.model_key = model_key
        self.display_name = MODELS[model_key]["display_name"]
        self.device = device
        self.directory = directory

        self.tokenizer = AutoTokenizer.from_pretrained(directory)

        self.model = load_model(model_key, directory).to(device)

        # eval() also switches MobileBERT's pooled BatchNorm to its
        # running statistics, so a single sentence classifies the
        # same way it would inside a batch.
        self.model.eval()

        # Fitted on validation during training. Applied here so every
        # consumer - CLI, server, fusion - sees the same calibrated
        # probabilities rather than each applying its own correction.
        self.temperature = load_temperature(directory)

    # --------------------------------------------------------

    def _to_shared(self, probabilities):
        """Project a task-space distribution onto the shared seven."""

        shared = {label: 0.0 for label in SHARED_LABELS}

        if self.task.key == "goemotions":
            # Already in the shared space, same order.
            for index, label in self.task.id2label.items():
                shared[label] = float(probabilities[index])

            return shared

        # Six-class MTEB: 'love' and 'joy' both land on 'happy', so
        # the mass is summed rather than overwritten.
        for index, label in self.task.id2label.items():
            shared[TEXT_TO_SHARED_LABEL[label]] += float(probabilities[index])

        return shared

    @torch.no_grad()
    def predict(self, text):
        """Predicted emotion, confidence, timing and both distributions."""

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MODELS[self.model_key]["max_length"],
            return_tensors="pt",
        ).to(self.device)

        started = time.perf_counter()

        logits = self.model(**encoded).logits

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        probabilities = F.softmax(logits / self.temperature, dim=-1)[0]

        index = int(probabilities.argmax())

        label = self.task.id2label[index]

        shared = self._to_shared(probabilities)

        # Token count excludes [CLS]/[SEP]. Fusion uses it as the
        # informativeness factor, so it has to measure the message
        # rather than the framing the tokenizer adds.
        token_count = max(0, int(encoded["input_ids"].shape[-1]) - 2)

        return {
            "text": text,
            "label": label,
            "confidence": float(probabilities[index]),
            "shared_label": max(shared, key=lambda key: shared[key]),
            "shared_distribution": shared,
            "token_count": token_count,
            "latency_ms": elapsed_ms,
            "task": self.task.key,
            "distribution": {
                self.task.id2label[i]: float(probabilities[i])
                for i in range(self.task.num_labels)
            },
        }


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def print_prediction(prediction, show_distribution=False):

    print(f'  "{prediction["text"]}"')

    fusion_note = (
        "" if prediction["label"] == prediction["shared_label"]
        else f"  [fusion: {prediction['shared_label']}]"
    )

    print(f"  -> {prediction['label']} "
          f"({prediction['confidence']*100:.1f}%)"
          f"{fusion_note}  "
          f"{prediction['token_count']} tok  "
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

    print(f"Model: {classifier.display_name}  "
          f"(task {classifier.task.key}, T={classifier.temperature:.3f})\n")

    # The demo sentences are written against the MTEB six-class
    # vocabulary. Under the shared seven, 'sadness' is spelled 'sad'
    # and both 'joy' and 'love' become 'happy', so the expected label
    # is translated before it is compared.
    expected_map = {
        "sadness": "sad", "joy": "happy", "love": "happy",
        "anger": "angry", "fear": "fear", "surprise": "surprise",
    }

    correct = 0

    for text, expected in EXAMPLES:

        prediction = classifier.predict(text)

        if classifier.task.key == "goemotions":
            expected = expected_map[expected]

        matched = prediction["label"] == expected

        correct += matched

        print(f"[{'ok ' if matched else 'MISS'}] expected {expected}")

        print_prediction(prediction, show_distribution)

    print(f"{correct}/{len(EXAMPLES)} demo sentences matched the intended emotion.")


def run_interactive(classifier, show_distribution):

    print(f"Model: {classifier.display_name}  "
          f"(task {classifier.task.key}, T={classifier.temperature:.3f})")
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
        "--task",
        default=DEFAULT_TASK,
        choices=list(TASKS),
        help=(
            "Which label space to predict in. 'goemotions' is the "
            "7-class shared space used by fusion and the web UI."
        ),
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

    classifier = TextEmotionClassifier(
        arguments.model, arguments.device, arguments.task
    )

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
