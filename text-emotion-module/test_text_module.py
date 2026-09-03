# ============================================================
# TEXT EMOTION MODULE - TESTS
# ============================================================
#
#   python test_text_module.py
#
# Also runs under pytest if it is installed:
#
#   pytest test_text_module.py -v
#
# Tests that need a trained checkpoint are skipped when the
# checkpoints directory is empty, so the suite is useful before
# training as well as after.
# ============================================================

import json
import sys
import traceback

import numpy as np
import torch
from transformers import AutoTokenizer

import config
from data import class_weights, collate, encode_split, load_splits
from metrics import compute_confusion_matrix, compute_metrics, format_report


class Skip(Exception):
    """Raised to mark a test as skipped rather than failed."""


def require_checkpoint(model_key):

    directory = config.checkpoint_path(model_key)

    if not (directory / "config.json").exists():
        raise Skip(f"no checkpoint for {model_key}")

    return directory


# ============================================================
# CONFIGURATION
# ============================================================

def test_label_maps_are_consistent():

    assert config.NUM_LABELS == 6

    assert set(config.ID2LABEL) == set(range(6))

    for index, name in config.ID2LABEL.items():
        assert config.LABEL2ID[name] == index

    # Every text label must have a bridge into the shared
    # vocabulary, or the fusion module will raise a KeyError.
    assert set(config.TEXT_TO_SHARED_LABEL) == set(config.ID2LABEL.values())


def test_shared_labels_are_a_subset_of_the_vision_labels():

    vision_labels = {
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    }

    assert set(config.TEXT_TO_SHARED_LABEL.values()) <= vision_labels


def test_heavy_models_are_excluded():

    # The project brief drops BERT-base and RoBERTa on size grounds.
    for settings in config.MODELS.values():
        hub_id = settings["hub_id"].lower()
        assert "roberta" not in hub_id
        assert hub_id != "bert-base-uncased"


# ============================================================
# DATA
# ============================================================

def test_splits_have_the_expected_shape():

    dataset = load_splits()

    assert len(dataset["train"]) == 15956
    assert len(dataset["validation"]) == 1988
    assert len(dataset["test"]) == 1986

    assert set(dataset["train"]["label"]) == set(range(6))


def test_collate_pads_to_the_longest_row_only():

    batch = [([1, 2, 3], 0), ([4, 5], 1), ([6], 2)]

    input_ids, attention_mask, labels = collate(batch, pad_token_id=0)

    assert input_ids.shape == (3, 3)

    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]

    assert labels.tolist() == [0, 1, 2]

    # Padded positions must be masked out, otherwise the model
    # attends to filler tokens.
    assert input_ids[attention_mask == 0].eq(0).all()


def test_encoded_split_lengths_match_the_source():

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    dataset = load_splits()

    encoded = encode_split(dataset["test"], tokenizer, 128)

    assert len(encoded) == len(dataset["test"])

    assert encoded.labels == list(dataset["test"]["label"])

    # Nothing should be truncated at 128 - the longest text is 87 tokens.
    assert max(len(ids) for ids in encoded.input_ids) < 128


def test_class_weights_favour_the_rare_classes():

    class Stub:
        labels = [0] * 90 + [1] * 10

    class StubLoader:
        dataset = Stub()

    weights = class_weights(StubLoader())

    assert weights.shape == (6,)

    # The rare class must outweigh the common one.
    assert weights[1] > weights[0]

    # Classes absent from the subset must not produce inf or nan.
    assert torch.isfinite(weights).all()


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def test_only_mobilebert_gets_the_pooled_normalisation():

    import modeling

    assert modeling.NEEDS_POOLED_NORMALIZATION == {"mobilebert"}


def test_mobilebert_logits_are_finite_and_input_dependent():

    import torch
    from transformers import AutoTokenizer

    from modeling import create_model

    torch.manual_seed(config.SEED)

    model = create_model("mobilebert")
    model.train()  # BatchNorm needs batch statistics here

    tokenizer = AutoTokenizer.from_pretrained(config.MODELS["mobilebert"]["hub_id"])

    encoded = tokenizer(
        [
            "i feel so happy today",
            "i am terrified of tomorrow",
            "i adore her deeply",
            "i am furious at them",
        ],
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    # Without the BatchNorm the raw pooled state is ~1e7 and the
    # initial logits land around 1e6, which makes the starting
    # cross-entropy ~6.7e6 instead of ln(6)=1.79.
    assert torch.isfinite(logits).all()

    assert logits.abs().max() < 100, f"logits exploded: {logits.abs().max()}"

    # And the four sentences must not collapse to one vector, which
    # is what happens with classifier_activation=True (tanh saturates).
    spread = logits.std(dim=0).max()

    assert spread > 1e-3, f"logits are input-independent (spread {spread})"


def test_adapted_head_survives_a_save_load_round_trip():

    import tempfile
    from pathlib import Path

    import torch

    from modeling import create_model, load_model

    torch.manual_seed(config.SEED)

    model = create_model("mobilebert")
    model.eval()

    inputs = {
        "input_ids": torch.randint(1000, 2000, (2, 12)),
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
    }

    with torch.no_grad():
        before = model(**inputs).logits

    with tempfile.TemporaryDirectory() as directory:

        model.save_pretrained(directory)

        reloaded = load_model("mobilebert", Path(directory))
        reloaded.eval()

        with torch.no_grad():
            after = reloaded(**inputs).logits

    # If load_model rebuilt a plain Linear head, the saved
    # classifier weights would be silently dropped and this would
    # differ.
    assert torch.allclose(before, after, atol=1e-6)


# ============================================================
# DEMO EXAMPLES
# ============================================================

def test_demo_sentences_are_not_in_the_dataset():

    from predict import EXAMPLES

    dataset = load_splits()

    seen = set()

    for split in dataset.values():
        seen.update(text.strip().lower() for text in split["text"])

    # The README states these are unseen phrasing rather than
    # sampled rows. If one ever leaks in from the dataset, the
    # demo would quietly become a training-set readout.
    for text, _ in EXAMPLES:
        assert text.strip().lower() not in seen, f"'{text}' is in the dataset"


def test_demo_covers_every_emotion():

    from predict import EXAMPLES

    covered = {emotion for _, emotion in EXAMPLES}

    assert covered == set(config.ID2LABEL.values())


# ============================================================
# METRICS
# ============================================================

def test_perfect_predictions_score_one():

    labels = np.array([0, 1, 2, 3, 4, 5])

    metrics = compute_metrics(labels, labels.copy())

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_majority_class_collapse_is_visible_in_macro_f1():

    # 100 rows, 90 of class 0. Predicting only class 0 gives high
    # accuracy but poor macro-F1 - this is exactly the failure the
    # headline metric is chosen to expose.
    labels = np.array([0] * 90 + [1] * 10)

    predictions = np.zeros(100, dtype=int)

    metrics = compute_metrics(labels, predictions)

    assert metrics["accuracy"] == 0.9

    assert metrics["macro_f1"] < 0.2


def test_confusion_matrix_is_square_and_totals_the_rows():

    labels = np.array([0, 0, 1, 2, 3, 4, 5])
    predictions = np.array([0, 1, 1, 2, 3, 4, 5])

    matrix = compute_confusion_matrix(labels, predictions)

    assert matrix.shape == (6, 6)

    assert matrix.sum() == len(labels)

    assert matrix[0].tolist() == [1, 1, 0, 0, 0, 0]


def test_report_renders_every_emotion():

    labels = np.array([0, 1, 2, 3, 4, 5])

    report = format_report(compute_metrics(labels, labels.copy()))

    for name in config.ID2LABEL.values():
        assert name in report


# ============================================================
# TRAINED MODELS
# ============================================================

def test_checkpoints_predict_in_range():

    from predict import TextEmotionClassifier

    trained = [
        key for key in config.MODELS
        if (config.checkpoint_path(key) / "config.json").exists()
    ]

    if not trained:
        raise Skip("no checkpoints trained yet")

    for key in trained:

        classifier = TextEmotionClassifier(key)

        prediction = classifier.predict("i am so happy to see you again")

        assert prediction["label"] in config.ID2LABEL.values()

        assert 0.0 <= prediction["confidence"] <= 1.0

        assert abs(sum(prediction["distribution"].values()) - 1.0) < 1e-4

        assert prediction["shared_label"] in config.TEXT_TO_SHARED_LABEL.values()


def test_predictions_are_deterministic():

    from predict import TextEmotionClassifier

    require_checkpoint("tinybert")

    classifier = TextEmotionClassifier("tinybert")

    sentence = "i am terrified of what happens next"

    first = classifier.predict(sentence)
    second = classifier.predict(sentence)

    assert first["label"] == second["label"]

    assert abs(first["confidence"] - second["confidence"]) < 1e-6


def test_batching_does_not_change_predictions():

    from predict import TextEmotionClassifier

    require_checkpoint("tinybert")

    classifier = TextEmotionClassifier("tinybert")

    sentences = [
        "i am so happy to see you again",
        "i feel completely alone tonight",
        "this makes me absolutely furious",
    ]

    one_at_a_time = [classifier.predict(s)["label"] for s in sentences]

    encoded = classifier.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = classifier.model(**encoded).logits

    batched = [config.ID2LABEL[int(i)] for i in logits.argmax(dim=-1)]

    # Padding must not leak into the result. If this fails, the
    # attention mask is not being applied correctly.
    assert one_at_a_time == batched


def test_benchmark_results_are_well_formed():

    if not config.RESULTS_FILE.exists():
        raise Skip("benchmark.py has not been run")

    with open(config.RESULTS_FILE) as handle:
        payload = json.load(handle)

    assert len(payload["results"]) == len(config.MODELS)

    for result in payload["results"]:

        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
        assert 0.0 <= result["metrics"]["macro_f1"] <= 1.0

        assert result["size"]["parameters"] > 0
        assert result["latency"]["median_ms"] > 0

        # Every candidate must genuinely be lighter than BERT-base.
        assert result["size"]["parameters"] < config.BERT_BASE_PARAMS

        matrix = np.array(result["confusion_matrix"])

        assert matrix.shape == (6, 6)
        assert matrix.sum() == 1986


def test_tinybert_is_the_smallest_of_the_three():

    if not config.RESULTS_FILE.exists():
        raise Skip("benchmark.py has not been run")

    with open(config.RESULTS_FILE) as handle:
        payload = json.load(handle)

    by_size = {r["model_key"]: r["size"]["parameters"] for r in payload["results"]}

    assert by_size["tinybert"] == min(by_size.values())

    # The headline claim from the TinyBERT paper: ~7x smaller
    # than BERT-base. Verify it holds for this checkpoint.
    ratio = config.BERT_BASE_PARAMS / by_size["tinybert"]

    assert ratio > 7.0


# ============================================================
# RUNNER
# ============================================================

def main():

    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    passed = failed = skipped = 0

    for name, function in tests:

        try:
            function()

        except Skip as reason:
            print(f"SKIP  {name}  ({reason})")
            skipped += 1

        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1

        else:
            print(f"PASS  {name}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
