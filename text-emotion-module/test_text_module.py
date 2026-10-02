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


def require_checkpoint(model_key, task="mteb"):

    directory = config.get_task(task).checkpoint_path(model_key)

    if not (directory / "config.json").exists():
        raise Skip(f"no {task} checkpoint for {model_key}")

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
    """Every trained checkpoint, in either task, predicts sanely.

    Both label spaces are covered rather than just the default, because
    the six-class checkpoints stay in the repository as the
    architecture comparison and must keep working.
    """

    from predict import TextEmotionClassifier

    trained = [
        (task_key, model_key)
        for task_key in config.TASKS
        for model_key in config.MODELS
        if (config.get_task(task_key).checkpoint_path(model_key)
            / "config.json").exists()
    ]

    if not trained:
        raise Skip("no checkpoints trained yet")

    for task_key, model_key in trained:

        task = config.get_task(task_key)

        classifier = TextEmotionClassifier(model_key, task=task_key)

        prediction = classifier.predict("i am so happy to see you again")

        assert prediction["task"] == task_key

        # The native label belongs to that task's own vocabulary.
        assert prediction["label"] in task.id2label.values()

        assert 0.0 <= prediction["confidence"] <= 1.0

        assert abs(sum(prediction["distribution"].values()) - 1.0) < 1e-4

        # The shared projection always lands in the seven-class space,
        # whichever task produced it - that is the contract fusion
        # depends on.
        assert prediction["shared_label"] in config.SHARED_LABELS

        assert set(prediction["shared_distribution"]) == set(config.SHARED_LABELS)

        assert abs(
            sum(prediction["shared_distribution"].values()) - 1.0
        ) < 1e-4

        assert prediction["token_count"] > 0

        if task_key == "mteb":
            # A six-class model cannot argue for either of the two
            # classes it was never trained on. This is the concrete
            # handicap that motivated the GoEmotions task.
            assert prediction["shared_distribution"]["neutral"] == 0.0
            assert prediction["shared_distribution"]["disgust"] == 0.0


def test_predictions_are_deterministic():

    from predict import TextEmotionClassifier

    require_checkpoint("tinybert")

    classifier = TextEmotionClassifier("tinybert", task="mteb")

    sentence = "i am terrified of what happens next"

    first = classifier.predict(sentence)
    second = classifier.predict(sentence)

    assert first["label"] == second["label"]

    assert abs(first["confidence"] - second["confidence"]) < 1e-6


def test_batching_does_not_change_predictions():

    from predict import TextEmotionClassifier

    require_checkpoint("tinybert")

    classifier = TextEmotionClassifier("tinybert", task="mteb")

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
# GOEMOTIONS TASK
# ============================================================
#
# These cover the label projection rather than the model. A
# silent error in the mapping would not crash anything - it
# would just train a perfectly healthy classifier on quietly
# wrong labels, and every metric downstream would look fine.
# ============================================================

def test_ekman_grouping_covers_every_goemotions_label():

    assert len(config.GOEMOTIONS_LABELS) == 28

    covered = set(config.GOEMOTIONS_TO_SHARED)

    assert covered == set(config.GOEMOTIONS_LABELS), (
        f"unmapped: {set(config.GOEMOTIONS_LABELS) - covered}"
    )


def test_ekman_groups_are_disjoint():

    seen = set()

    for group, members in config.EKMAN_TO_SHARED.items():
        for member in members:
            assert member not in seen, f"{member} appears in two groups"
            seen.add(member)


def test_ekman_groups_land_on_the_shared_space():

    targets = set(config.GOEMOTIONS_TO_SHARED.values())

    assert targets == set(config.SHARED_LABELS), (
        f"missing shared labels: {set(config.SHARED_LABELS) - targets}"
    )

    # The two classes that motivated the whole task must be
    # reachable, or fusion is back to five usable classes.
    assert config.GOEMOTIONS_TO_SHARED["neutral"] == "neutral"
    assert config.GOEMOTIONS_TO_SHARED["disgust"] == "disgust"


def test_goemotions_id_mapping_matches_names():

    for index, name in enumerate(config.GOEMOTIONS_LABELS):

        expected = config.SHARED_LABEL2ID[config.GOEMOTIONS_TO_SHARED[name]]

        assert config.GOEMOTIONS_ID_TO_SHARED_ID[index] == expected


def test_specific_ekman_assignments():

    # Spot-checks of the groupings a reader is most likely to
    # question, pinned so a future edit has to be deliberate.
    cases = {
        "annoyance": "angry",
        "disapproval": "angry",
        "nervousness": "fear",
        "gratitude": "happy",
        "pride": "happy",
        "love": "happy",
        "grief": "sad",
        "remorse": "sad",
        "embarrassment": "sad",
        "curiosity": "surprise",
        "confusion": "surprise",
        "realization": "surprise",
    }

    for fine, shared in cases.items():
        assert config.GOEMOTIONS_TO_SHARED[fine] == shared, (
            f"{fine} -> {config.GOEMOTIONS_TO_SHARED[fine]}, expected {shared}"
        )


def test_shared_space_matches_the_deployment_module():
    """The two copies of the shared vocabulary must agree.

    config.SHARED_LABELS is duplicated from the deployment folder so
    this module stays runnable alone. The ONNX classifier head is the
    real source of truth, and a divergence would misalign every
    probability vector fusion compares.
    """

    from pathlib import Path

    deployment = Path(__file__).resolve().parent.parent / "vit-emotion-v2-deployment"

    if not (deployment / "labels.py").exists():
        raise Skip("deployment module not present")

    sys.path.insert(0, str(deployment))

    try:
        import labels as deployment_labels
    finally:
        sys.path.remove(str(deployment))

    assert tuple(config.SHARED_LABELS) == tuple(deployment_labels.SHARED_LABELS)

    # And both must match what the exported model actually emits.
    emotion_config = json.loads(
        (deployment / "emotion_config.json").read_text(encoding="utf-8")
    )

    ordered = tuple(
        emotion_config["id2label"][str(index)]
        for index in range(len(emotion_config["id2label"]))
    )

    assert ordered == tuple(config.SHARED_LABELS)


def test_task_registry_keeps_the_tasks_apart():

    mteb = config.get_task("mteb")
    goemotions = config.get_task("goemotions")

    assert mteb.num_labels == 6
    assert goemotions.num_labels == 7

    # Distinct checkpoint directories, or training one task would
    # overwrite the other's weights.
    assert mteb.checkpoint_path("tinybert") != goemotions.checkpoint_path("tinybert")
    assert mteb.results_file != goemotions.results_file

    # The MTEB task must keep its original layout so the already
    # trained six-class checkpoints stay discoverable.
    assert mteb.checkpoint_path("tinybert").name == "tinybert"

    try:
        config.get_task("nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown task should raise ValueError")


# ============================================================
# METRICS ACROSS LABEL SPACES
# ============================================================

def test_metrics_respect_the_task_label_space():

    labels = np.array([0, 1, 2, 3, 4, 5, 6])

    result = compute_metrics(labels, labels, config.SHARED_ID2LABEL)

    assert result["accuracy"] == 1.0
    assert set(result["per_class"]) == set(config.SHARED_LABELS)

    matrix = compute_confusion_matrix(labels, labels, config.SHARED_ID2LABEL)
    assert matrix.shape == (7, 7)

    # Default still scores in the six-class MTEB space.
    six = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert set(six["per_class"]) == set(config.ID2LABEL.values())


# ============================================================
# CALIBRATION
# ============================================================

def test_temperature_scaling_reduces_ece_without_changing_predictions():

    from calibration import (
        expected_calibration_error,
        fit_temperature,
        softmax,
    )

    rng = np.random.default_rng(0)

    count, classes = 3000, 7

    labels = rng.integers(0, classes, size=count)
    logits = rng.normal(0.0, 1.0, size=(count, classes))

    hit = rng.random(count) < 0.7
    logits[np.arange(count)[hit], labels[hit]] += 3.0

    # Inflate so the model is overconfident, the condition
    # temperature scaling exists to correct.
    logits *= 2.5

    temperature, before, after = fit_temperature(logits, labels)

    assert after < before, f"ECE got worse: {before} -> {after}"

    # Scaling is monotone, so it must not move a single argmax.
    assert np.array_equal(
        logits.argmax(axis=1), (logits / temperature).argmax(axis=1)
    )

    assert abs(softmax(logits / temperature).sum(axis=1) - 1.0).max() < 1e-9


def test_missing_calibration_file_falls_back_to_one():
    """Checkpoints trained before calibration existed must still load."""

    from pathlib import Path

    from calibration import load_temperature

    assert load_temperature(Path("does-not-exist")) == 1.0


# ============================================================
# GOEMOTIONS DATA (network)
# ============================================================

def test_goemotions_projection_is_single_label():
    """Every surviving row must carry exactly one shared label."""

    from data import _to_shared_label

    # A row whose fine labels all collapse to one group survives.
    admiration = config.GOEMOTIONS_LABELS.index("admiration")
    gratitude = config.GOEMOTIONS_LABELS.index("gratitude")

    assert _to_shared_label([admiration, gratitude]) == config.SHARED_LABEL2ID["happy"]

    # A row spanning two groups is genuinely ambiguous and dropped.
    sadness = config.GOEMOTIONS_LABELS.index("sadness")
    assert _to_shared_label([admiration, sadness]) is None

    # Single labels pass through.
    neutral = config.GOEMOTIONS_LABELS.index("neutral")
    assert _to_shared_label([neutral]) == config.SHARED_LABEL2ID["neutral"]


def test_class_weights_handle_seven_classes():

    from data import task_class_weights

    class Stub:
        labels = [0, 0, 0, 0, 1, 2, 3, 3, 4, 5, 6]

    class Loader:
        dataset = Stub()

    weights = task_class_weights(Loader(), 7)

    assert len(weights) == 7
    assert torch.all(weights > 0)

    # The rarest classes must be weighted above the commonest.
    assert weights[1] > weights[0]


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
