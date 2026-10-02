# ============================================================
# TEXT EMOTION MODULE - TRAINING
# ============================================================
#
# Fine-tunes one compact transformer on MTEB EmotionClassification.
#
#   python train.py --model tinybert
#   python train.py --model distilbert
#   python train.py --model mobilebert
#   python train.py --model all
#
# A plain PyTorch loop is used instead of transformers.Trainer
# so the training procedure stays identical and visible across
# all three architectures.
# ============================================================

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from calibration import fit_temperature
from config import (
    DEFAULT_TASK,
    MAX_GRAD_NORM,
    MODELS,
    SEED,
    TASKS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    ensure_directories,
    get_task,
)
from data import build_task_dataloaders, task_class_weights
from metrics import compute_metrics
from modeling import create_model, load_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# EVALUATION PASS
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, task):
    """Run the model over a loader.

    Returns (metrics, predictions, labels, logits). The raw logits come
    back too because temperature calibration needs them, and running a
    second full pass just to collect them would double the cost of
    every epoch for no reason.
    """

    model.eval()

    all_logits = []
    all_labels = []

    for input_ids, attention_mask, labels in loader:

        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    predictions = logits.argmax(axis=-1)

    return (
        compute_metrics(labels, predictions, task.id2label),
        predictions,
        labels,
        logits,
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(model_key, task, max_train_samples=None, device="cpu"):

    settings = MODELS[model_key]

    print("=" * 60)
    print(f"TRAINING  {settings['display_name']}")
    print(f"Backbone  {settings['hub_id']}")
    print(f"Task      {task.key}  ({task.num_labels} classes)")
    print(f"Dataset   {task.dataset_id}")
    print("=" * 60)

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(settings["hub_id"])

    model = create_model(model_key, task).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Parameters {parameter_count/1e6:.1f}M")

    train_loader, validation_loader, test_loader = build_task_dataloaders(
        task,
        tokenizer,
        settings["batch_size"],
        settings["max_length"],
        max_train_samples=max_train_samples,
    )

    print(f"Train batches {len(train_loader)}  "
          f"Val batches {len(validation_loader)}  "
          f"Test batches {len(test_loader)}")

    # --------------------------------------------------------
    # OPTIMISER
    # --------------------------------------------------------

    no_decay = ("bias", "LayerNorm.weight")

    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(grouped_parameters, lr=settings["learning_rate"])

    total_steps = len(train_loader) * settings["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=task_class_weights(train_loader, task.num_labels).to(device)
    )

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    history = []

    best_macro_f1 = -1.0

    started = time.time()

    for epoch in range(1, settings["epochs"] + 1):

        model.train()

        running_loss = 0.0
        epoch_started = time.time()

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, 1):

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits

            loss = loss_function(logits, labels.to(device))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_started
                print(f"  epoch {epoch}  step {step}/{len(train_loader)}  "
                      f"loss {running_loss/step:.4f}  "
                      f"{elapsed/step:.2f}s/step",
                      flush=True)

        validation_metrics, _, _, _ = evaluate(
            model, validation_loader, device, task
        )

        print(f"  epoch {epoch} validation  "
              f"accuracy {validation_metrics['accuracy']:.4f}  "
              f"macro-F1 {validation_metrics['macro_f1']:.4f}",
              flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        })

        # Keep the best epoch by macro-F1, not accuracy. On MTEB
        # accuracy is dominated by 'joy' and 'sadness' (63% of the
        # data); on GoEmotions by 'happy' and 'neutral' (71%). Either
        # way accuracy would hide collapse on the rare classes.
        if validation_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = validation_metrics["macro_f1"]

            destination = task.checkpoint_path(model_key)
            destination.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)

            print(f"  saved checkpoint (best macro-F1 {best_macro_f1:.4f})",
                  flush=True)

    training_seconds = time.time() - started

    # --------------------------------------------------------
    # TEST THE BEST CHECKPOINT
    # --------------------------------------------------------

    best_model = load_model(model_key, task.checkpoint_path(model_key)).to(device)

    test_metrics, predictions, labels, _ = evaluate(
        best_model, test_loader, device, task
    )

    print(f"\nTEST  accuracy {test_metrics['accuracy']:.4f}  "
          f"macro-F1 {test_metrics['macro_f1']:.4f}")

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------
    #
    # Fitted on validation, never on test. The class-weighted loss
    # above deliberately distorts the decision boundary toward the
    # rare classes, and that leaves the probabilities miscalibrated as
    # a side effect. Fusion consumes those probabilities directly, so
    # an overconfident modality would win arguments it has not earned.

    _, _, validation_labels, validation_logits = evaluate(
        best_model, validation_loader, device, task
    )

    temperature, ece_before, ece_after = fit_temperature(
        validation_logits, validation_labels
    )

    print(f"CALIBRATION  temperature {temperature:.3f}  "
          f"ECE {ece_before:.4f} -> {ece_after:.4f}\n")

    record = {
        "model_key": model_key,
        "task": task.key,
        "dataset_id": task.dataset_id,
        "labels": task.labels,
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "hub_id": settings["hub_id"],
        "display_name": settings["display_name"],
        "parameters": parameter_count,
        "epochs": settings["epochs"],
        "learning_rate": settings["learning_rate"],
        "batch_size": settings["batch_size"],
        "max_length": settings["max_length"],
        "train_samples": len(train_loader.dataset),
        "training_seconds": training_seconds,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "test": test_metrics,
    }

    destination = task.checkpoint_path(model_key)

    with open(destination / "training_summary.json", "w") as handle:
        json.dump(record, handle, indent=2)

    # Stored beside the weights so predict.py and the server can apply
    # the same temperature without re-deriving it.
    with open(destination / "calibration.json", "w") as handle:
        json.dump(
            {
                "temperature": temperature,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "fitted_on": "validation",
            },
            handle,
            indent=2,
        )

    np.save(destination / "test_predictions.npy", predictions)
    np.save(destination / "test_labels.npy", labels)

    return record


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODELS) + ["all"],
        help="Which model to fine-tune.",
    )

    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        choices=list(TASKS),
        help=(
            "Which dataset and label space. 'goemotions' is the 7-class "
            "space the fusion module consumes; 'mteb' is the original "
            "6-class architecture comparison."
        ),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Truncate the training split (smoke tests only).",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    arguments = parser.parse_args()

    ensure_directories()

    task = get_task(arguments.task)

    keys = list(MODELS) if arguments.model == "all" else [arguments.model]

    for key in keys:
        train_model(key, task, arguments.max_train_samples, arguments.device)

    print(
        f"Training complete. Run 'python benchmark.py --task {task.key}' "
        f"to compare."
    )


if __name__ == "__main__":
    main()
