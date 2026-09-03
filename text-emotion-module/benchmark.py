# ============================================================
# TEXT EMOTION MODULE - BENCHMARK
# ============================================================
#
# Compares the three fine-tuned models on the MTEB emotion test
# split along the three axes that matter for this project:
#
#   1. Quality  - accuracy, macro-F1, per-class F1
#   2. Size     - parameters, on-disk megabytes
#   3. Speed    - single-sentence CPU latency, batched throughput
#
# Run after train.py:
#
#   python benchmark.py
#
# Writes results/benchmark.json, which report.py turns into the
# comparison charts used in the README.
# ============================================================

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    NUM_LABELS,
    BERT_BASE_PARAMS,
    MODELS,
    RESULTS_FILE,
    checkpoint_path,
    ensure_directories,
)
from data import build_dataloaders
from metrics import compute_confusion_matrix, compute_metrics, format_report
from modeling import load_model
from train import evaluate


# ============================================================
# LATENCY
# ============================================================
#
# Latency is measured on a single sentence because that is how
# the module is actually called at inference time - one user
# utterance at a time, alongside one webcam frame.
# ============================================================

LATENCY_SENTENCE = "i feel so grateful that you stayed with me through all of it"

WARMUP_RUNS = 10
TIMED_RUNS = 100


@torch.no_grad()
def measure_latency(model, tokenizer, device):
    """Median and p95 milliseconds for one forward pass on one sentence."""

    model.eval()

    encoded = tokenizer(LATENCY_SENTENCE, return_tensors="pt").to(device)

    for _ in range(WARMUP_RUNS):
        model(**encoded)

    timings = []

    for _ in range(TIMED_RUNS):

        started = time.perf_counter()

        model(**encoded)

        timings.append((time.perf_counter() - started) * 1000.0)

    timings = np.array(timings)

    return {
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(timings.mean()),
    }


# ============================================================
# SIZE
# ============================================================

def measure_size(model_key, model):
    """Parameter count and the size of the saved weight files."""

    parameters = sum(p.numel() for p in model.parameters())

    weight_bytes = sum(
        path.stat().st_size
        for path in checkpoint_path(model_key).glob("*")
        if path.suffix in (".safetensors", ".bin")
    )

    return {
        "parameters": parameters,
        "disk_mb": weight_bytes / (1024 * 1024),
        "smaller_than_bert_base": BERT_BASE_PARAMS / parameters,
    }


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(model_key, device):

    settings = MODELS[model_key]

    directory = checkpoint_path(model_key)

    if not directory.exists():
        raise FileNotFoundError(
            f"No checkpoint at {directory}. Run: python train.py --model {model_key}"
        )

    print("=" * 60)
    print(f"BENCHMARKING  {settings['display_name']}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(directory)

    model = load_model(model_key, directory).to(device)

    _, _, test_loader = build_dataloaders(
        tokenizer,
        settings["batch_size"],
        settings["max_length"],
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    throughput_started = time.perf_counter()

    metrics, predictions, labels = evaluate(model, test_loader, device)

    throughput_seconds = time.perf_counter() - throughput_started

    print(format_report(metrics))

    # --------------------------------------------------------
    # SIZE AND SPEED
    # --------------------------------------------------------

    size = measure_size(model_key, model)

    latency = measure_latency(model, tokenizer, device)

    print(f"\nParameters      {size['parameters']/1e6:.1f}M "
          f"({size['smaller_than_bert_base']:.1f}x smaller than BERT-base)")
    print(f"On disk         {size['disk_mb']:.1f} MB")
    print(f"Latency         {latency['median_ms']:.1f} ms median, "
          f"{latency['p95_ms']:.1f} ms p95")
    print(f"Test throughput {len(labels)/throughput_seconds:.1f} sentences/s\n")

    # Training time is recovered from the run that produced the
    # checkpoint rather than re-measured here.
    summary_path = directory / "training_summary.json"

    training_seconds = None

    if summary_path.exists():
        with open(summary_path) as handle:
            training_seconds = json.load(handle)["training_seconds"]

    return {
        "model_key": model_key,
        "display_name": settings["display_name"],
        "hub_id": settings["hub_id"],
        "note": settings["note"],
        "metrics": metrics,
        "size": size,
        "latency": latency,
        "throughput_sentences_per_second": len(labels) / throughput_seconds,
        "training_seconds": training_seconds,
        "confusion_matrix": compute_confusion_matrix(labels, predictions).tolist(),
    }


# ============================================================
# BERT-BASE REFERENCE
# ============================================================
#
# BERT-base is not trained here - it is the model the three
# candidates are distilled from, and the one this project drops
# on size grounds. Its parameter count and latency are measured
# anyway, untrained, so the "7x smaller and 9x faster" claim can
# be checked rather than repeated. Fine-tuning changes neither
# of those two numbers.
# ============================================================

BERT_BASE_ID = "bert-base-uncased"


def benchmark_reference(device):

    print("=" * 60)
    print("REFERENCE  BERT-base (untrained, for size/speed only)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE_ID,
        num_labels=NUM_LABELS,
    ).to(device)

    parameters = sum(p.numel() for p in model.parameters())

    latency = measure_latency(model, tokenizer, device)

    print(f"Parameters      {parameters/1e6:.1f}M")
    print(f"Latency         {latency['median_ms']:.1f} ms median\n")

    return {
        "hub_id": BERT_BASE_ID,
        "parameters": parameters,
        "latency": latency,
        "trained": False,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(results):

    print("=" * 78)
    print("COMPARISON  -  MTEB EmotionClassification test split (1986 rows)")
    print("=" * 78)

    header = (f"{'Model':<22} {'Params':>8} {'Disk':>8} {'Acc':>8} "
              f"{'MacroF1':>8} {'Latency':>9}")

    print(header)
    print("-" * 78)

    for result in results:
        print(
            f"{result['display_name']:<22} "
            f"{result['size']['parameters']/1e6:>7.1f}M "
            f"{result['size']['disk_mb']:>7.1f}M "
            f"{result['metrics']['accuracy']*100:>7.2f}% "
            f"{result['metrics']['macro_f1']*100:>7.2f}% "
            f"{result['latency']['median_ms']:>8.1f}ms"
        )

    print("-" * 78)

    best = max(results, key=lambda r: r["metrics"]["macro_f1"])
    fastest = min(results, key=lambda r: r["latency"]["median_ms"])
    smallest = min(results, key=lambda r: r["size"]["parameters"])

    print(f"Best macro-F1 : {best['display_name']}")
    print(f"Fastest       : {fastest['display_name']}")
    print(f"Smallest      : {smallest['display_name']}")
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--device",
        default="cpu",
        help="Latency is reported for CPU by default, matching deployment.",
    )

    arguments = parser.parse_args()

    ensure_directories()

    results = [
        benchmark_model(key, arguments.device)
        for key in MODELS
    ]

    reference = benchmark_reference(arguments.device)

    print_comparison(results)

    fastest = min(results, key=lambda r: r["latency"]["median_ms"])

    print(f"vs BERT-base  : {fastest['display_name']} is "
          f"{reference['parameters']/fastest['size']['parameters']:.1f}x smaller and "
          f"{reference['latency']['median_ms']/fastest['latency']['median_ms']:.1f}x faster")
    print("=" * 78)

    payload = {
        "dataset": "mteb/emotion (MTEB EmotionClassification)",
        "device": arguments.device,
        "torch_threads": torch.get_num_threads(),
        "bert_base_parameters": BERT_BASE_PARAMS,
        "bert_base_reference": reference,
        "results": results,
    }

    with open(RESULTS_FILE, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
