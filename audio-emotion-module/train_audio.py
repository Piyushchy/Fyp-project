# ============================================================
# AUDIO EMOTION MODULE - TRAINING
# ============================================================
#
# Trains one entry of config.MODELS on the rebuilt corpus and
# writes the best checkpoint by validation macro-F1.
#
#   python train_audio.py --model mfcc-cnn
#   python train_audio.py --model wavlm-base-plus
#
# Why macro-F1 selects the checkpoint, not accuracy
# -------------------------------------------------
# Validation is left at its natural distribution, where neutral
# is roughly a quarter of the split. Selecting on accuracy picks
# the epoch that is best at the majority classes, which is
# reliably the epoch that has most nearly given up on disgust
# and fear. The fusion needs a modality whose rare classes are
# worth listening to, so the rare classes have to count in the
# selection rule.
#
# Why two learning rates
# ----------------------
# The SSL head is randomly initialised and the backbone carries
# 94k hours of pretraining. A single rate either leaves the head
# undertrained or washes the backbone out in the first epoch;
# 1e-3 on the head and 5e-5 on the backbone is the standard
# 20x separation. The MFCC baseline has no backbone, so it runs
# one group and is unaffected.
#
# Why bf16 and not fp16
# ---------------------
# WavLM's attention logits and the layer-sum softmax both run
# hot enough to overflow fp16's range on this data, which shows
# up as a loss that goes NaN somewhere in epoch 1 and is the
# usual reason SSL fine-tunes are reported as unstable. bf16 has
# fp32's exponent range and needs no loss scaler. Ampere and
# later only; the script falls back to fp32 elsewhere.
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

import modeling
from config import (
    CLASS_WEIGHT_POWER,
    LABELS,
    LABEL_SMOOTHING,
    MAX_GRAD_NORM,
    MODELS,
    RESULTS_DIR,
    SEED_TRAIN,
    checkpoint_path,
    ensure_directories,
)
from data import build_loaders, class_weights, describe, load_manifest
from metrics import compute_metrics, confusion, format_confusion, format_report


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SCHEDULE
# ============================================================

def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """Linear warmup to 1.0, then cosine decay to 0.

    The warmup is what keeps the first few hundred steps from moving
    the pretrained backbone before the randomly-initialised head has
    produced a gradient worth following.
    """

    if step < warmup:
        return (step + 1) / max(1, warmup)

    progress = (step - warmup) / max(1, total - warmup)

    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: nn.Module, spec: dict):
    groups = [
        {
            "params": model.head_parameters(),
            "lr": spec["learning_rate"],
        }
    ]

    backbone = model.backbone_parameters()
    backbone_lr = spec.get("backbone_learning_rate")

    if backbone and backbone_lr:
        groups.append({"params": backbone, "lr": backbone_lr})

    return torch.optim.AdamW(groups, weight_decay=spec["weight_decay"])


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, amp_dtype):
    """Probabilities, predictions and labels for one loader."""

    model.eval()

    all_probabilities = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(features)

        # Softmax in fp32: under bf16 the 8-bit mantissa quantises the
        # small probabilities badly, and those are exactly the entries
        # the fusion's log-pool takes the logarithm of.
        probabilities = torch.softmax(logits.float(), dim=-1)

        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())

    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)

    return probabilities, probabilities.argmax(axis=1), labels


# ============================================================
# TRAINING
# ============================================================

def train(
    model_key: str,
    epochs: int | None,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
    log,
) -> dict:
    spec = dict(MODELS[model_key])

    if epochs is not None:
        spec["epochs"] = epochs

    if batch_size is not None:
        spec["batch_size"] = batch_size

    accumulation = int(spec.get("gradient_accumulation", 1))

    rows = load_manifest()

    log("\n--- corpus ---")
    log(describe(rows))

    train_loader, validation_loader, test_loader, splits = build_loaders(
        rows,
        kind=spec["kind"],
        batch_size=spec["batch_size"],
        num_workers=num_workers,
    )

    model = modeling.build(model_key).to(device)

    log(f"\n--- {spec['display_name']} ---")
    log(f"  parameters      {modeling.parameter_count(model):,}")
    log(f"  batch size      {spec['batch_size']} x {accumulation} accumulated")
    log(f"  epochs          {spec['epochs']}")
    log(f"  head lr         {spec['learning_rate']}")
    log(f"  backbone lr     {spec.get('backbone_learning_rate')}")

    weights = class_weights(splits["train"], CLASS_WEIGHT_POWER).to(device)

    log("  class weights   " + "  ".join(
        f"{name}:{value:.2f}" for name, value in zip(LABELS, weights.tolist())
    ))

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=LABEL_SMOOTHING
    )

    optimizer = build_optimizer(model, spec)

    steps_per_epoch = max(1, len(train_loader) // accumulation)
    total_steps = steps_per_epoch * spec["epochs"]
    warmup_steps = int(total_steps * spec["warmup_ratio"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(step, total_steps, warmup_steps),
    )

    # bf16 only where it is native. On anything older the autocast
    # would be emulated and slower than just running fp32.
    amp_dtype = None

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        log("  precision       bf16 autocast")
    else:
        log("  precision       fp32")

    destination = checkpoint_path(model_key)
    destination.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_gain = 0
    history = []

    started = time.time()

    for epoch in range(1, spec["epochs"] + 1):
        model.train()

        running_loss = 0.0
        seen = 0
        epoch_started = time.time()

        optimizer.zero_grad(set_to_none=True)

        for index, (features, labels) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(features)
                loss = criterion(logits.float(), labels)

            # Scale so the accumulated gradient is the mean over the
            # effective batch rather than its sum.
            (loss / accumulation).backward()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

            if (index + 1) % accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (index + 1) % 100 == 0:
                log(
                    f"    epoch {epoch} step {index + 1}/{len(train_loader)} "
                    f"loss {running_loss / max(1, seen):.4f}",
                    flush=True,
                )

        # A trailing partial accumulation group still holds gradients.
        # Dropping them silently discards up to accumulation-1 batches
        # of every epoch.
        if len(train_loader) % accumulation:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        probabilities, predictions, truth = evaluate(
            model, validation_loader, device, amp_dtype
        )
        scores = compute_metrics(truth, predictions, probabilities)

        elapsed = time.time() - epoch_started

        log(
            f"  epoch {epoch:2d}  loss {running_loss / max(1, seen):.4f}  "
            f"val acc {scores['accuracy']:.4f}  "
            f"val macro-F1 {scores['macro_f1']:.4f}  "
            f"ECE {scores['ece']:.4f}  ({elapsed:.0f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "val_accuracy": scores["accuracy"],
                "val_macro_f1": scores["macro_f1"],
                "val_ece": scores["ece"],
                "seconds": round(elapsed, 1),
            }
        )

        if scores["macro_f1"] > best_macro_f1:
            best_macro_f1 = scores["macro_f1"]
            best_epoch = epoch
            epochs_without_gain = 0

            torch.save(model.state_dict(), destination / "model.pt")

            log(f"    saved (val macro-F1 {best_macro_f1:.4f})")
        else:
            epochs_without_gain += 1

            if epochs_without_gain >= spec["patience"]:
                log(f"    no gain in {spec['patience']} epochs, stopping")
                break

    training_seconds = time.time() - started

    # Score the SELECTED checkpoint, not whatever the last epoch left
    # in memory. Without this reload an early-stopped run reports the
    # metrics of a model it did not keep.
    model.load_state_dict(
        torch.load(destination / "model.pt", map_location=device)
    )

    # Where the encoder found the emotion. Initialised uniform, so
    # anything that deviates from 1/13 is learned evidence about which
    # depth carries paralinguistic information in this backbone.
    layer_distribution = None

    if hasattr(model, "layer_sum"):
        layer_distribution = model.layer_sum.distribution()

        log("\n--- learned layer mixture ---")
        log("  " + "  ".join(
            f"L{index}:{weight:.3f}"
            for index, weight in enumerate(layer_distribution)
        ))

        peak = int(np.argmax(layer_distribution))
        log(f"  peak at layer {peak} of {len(layer_distribution) - 1}")

    log(f"\n--- test, best epoch {best_epoch} ---")

    probabilities, predictions, truth = evaluate(
        model, test_loader, device, amp_dtype
    )
    test_scores = compute_metrics(truth, predictions, probabilities)

    log(format_report(test_scores, "  overall"))
    log("\n  confusion (rows true, columns predicted)")
    log(format_confusion(confusion(truth, predictions)))

    # Per-domain. The studio corpora and MELD are different problems
    # and a single averaged number hides which one moved.
    domains = np.array([row["domain"] for row in splits["test"]])
    per_domain = {}

    for domain in sorted(set(domains.tolist())):
        mask = domains == domain

        if not mask.any():
            continue

        domain_scores = compute_metrics(
            truth[mask], predictions[mask], probabilities[mask]
        )
        per_domain[domain] = domain_scores

        log("\n" + format_report(domain_scores, f"  domain: {domain}"))

    summary = {
        "model": model_key,
        "display_name": spec["display_name"],
        "kind": spec["kind"],
        "hub_id": spec["hub_id"],
        "note": spec["note"],
        "parameters": modeling.parameter_count(model),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "training_seconds": round(training_seconds, 1),
        "epochs_run": len(history),
        "history": history,
        "test": test_scores,
        "test_by_domain": per_domain,
        "layer_mixture": layer_distribution,
        "hyperparameters": {
            key: value
            for key, value in spec.items()
            if key not in ("note", "display_name")
        },
        "class_weight_power": CLASS_WEIGHT_POWER,
        "label_smoothing": LABEL_SMOOTHING,
    }

    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # The raw test probabilities, kept so the fusion evaluation can
    # pool this modality without re-running the encoder.
    np.savez_compressed(
        destination / "test_predictions.npz",
        probabilities=probabilities,
        labels=truth,
        uids=np.array([row["uid"] for row in splits["test"]]),
        domains=domains,
    )

    return summary


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="wavlm-base-plus", choices=sorted(MODELS)
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    ensure_directories()
    seed_everything(SEED_TRAIN)

    if arguments.device:
        device = torch.device(arguments.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # .txt, not .log: the repo's root .gitignore drops *.log, and this
    # transcript is a result worth keeping. Matches the text module's
    # results/train_log.txt.
    log_path = RESULTS_DIR / f"train_{arguments.model}.txt"
    handle = log_path.open("w", encoding="utf-8")

    def log(message: str = "", flush: bool = False) -> None:
        print(message, flush=flush)
        handle.write(message + "\n")

        if flush:
            handle.flush()

    try:
        log("=" * 62)
        log(f"TRAINING {arguments.model}")
        log("=" * 62)
        log(f"  device  {device}")

        if device.type == "cuda":
            log(f"  gpu     {torch.cuda.get_device_name(0)}")

        summary = train(
            arguments.model,
            arguments.epochs,
            arguments.batch_size,
            arguments.num_workers,
            device,
            log,
        )

        log(
            f"\ndone: test macro-F1 {summary['test']['macro_f1']:.4f}, "
            f"accuracy {summary['test']['accuracy']:.4f}"
        )
    finally:
        handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
