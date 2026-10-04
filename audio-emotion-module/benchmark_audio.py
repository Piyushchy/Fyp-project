# ============================================================
# AUDIO EMOTION MODULE - BENCHMARK
# ============================================================
#
# Collects the trained models into one comparison table, times
# them, and publishes the winner's per-class competence to the
# fusion engine.
#
#   python benchmark_audio.py
#
# Three jobs, in order
# --------------------
# 1. Compare. The MFCC CNN is the multimodal branch's own
#    architecture retrained on the rebuilt corpus; WavLM is the
#    model meant for deployment. Reporting them side by side is
#    what separates "the data was the problem" from "the model
#    was the problem" - a single good number from WavLM alone
#    would not distinguish the two.
#
# 2. Time. The deployment target runs face inference every frame
#    and audio on a rolling window. An encoder that cannot keep
#    up with its own window length is not deployable whatever it
#    scores, so latency is reported next to accuracy rather than
#    in a footnote.
#
# 3. Publish. reliability.json is what the fusion engine reads.
#    Writing the measured per-class F1 there is what actually
#    connects this module to the pool - without it the engine
#    treats audio as uniformly competent across all seven
#    classes, which it very much is not.
# ============================================================

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import modeling
from config import (
    CLIP_SAMPLES,
    LABELS,
    MFCC_FRAMES,
    MODELS,
    N_MFCC,
    RESULTS_DIR,
    RESULTS_FILE,
    checkpoint_path,
)

RELIABILITY_PATH = (
    Path(__file__).resolve().parent.parent
    / "vit-emotion-v2-deployment"
    / "reliability.json"
)


# ============================================================
# LATENCY
# ============================================================

def measure_latency(
    model_key: str, device: torch.device, repeats: int = 50
) -> dict:
    """Single-clip inference time, warmed up.

    Batch size 1, because that is how a live session calls it: one
    rolling window at a time. A batched throughput figure would be
    several times better and would not describe the deployment.
    """

    spec = MODELS[model_key]
    model = modeling.build(model_key).to(device).eval()

    path = checkpoint_path(model_key) / "model.pt"

    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))

    if spec["kind"] == "ssl":
        sample = torch.randn(1, CLIP_SAMPLES, device=device)
    else:
        sample = torch.randn(1, 1, N_MFCC, MFCC_FRAMES, device=device)

    with torch.no_grad():
        # Warmup. The first CUDA call pays kernel autotuning and
        # allocator setup that no later call does, and including it
        # would roughly double the reported median.
        for _ in range(10):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []

        for _ in range(repeats):
            started = time.perf_counter()
            model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            timings.append((time.perf_counter() - started) * 1000.0)

    timings = np.array(timings)

    return {
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(timings.mean()),
        "device": str(device),
        "batch_size": 1,
    }


# ============================================================
# PUBLISHING TO THE FUSION ENGINE
# ============================================================

def publish_reliability(summary: dict, base_weight: float) -> None:
    """Write the audio profile into reliability.json.

    Preserves the rest of the file. The face and text entries were
    tuned against each other and nothing here has license to touch
    them.
    """

    if not RELIABILITY_PATH.exists():
        print(f"  {RELIABILITY_PATH} not found, skipping publish")
        return

    payload = json.loads(RELIABILITY_PATH.read_text(encoding="utf-8"))

    # The wild-domain numbers, not the overall ones. The deployment
    # input is a laptop microphone in a room, which is MELD's
    # condition and not the studio corpora's; publishing the blended
    # figure would tell the pool to trust audio more than it should
    # on exactly the input it will actually receive.
    wild = summary["test_by_domain"].get("wild", summary["test"])

    payload["audio"] = {
        "source": (
            f"{summary['display_name']} on the audio module's "
            f"speaker-disjoint test split, wild (MELD) domain only - "
            f"{wild['support']} clips. Overall across both domains: "
            f"{summary['test']['accuracy']:.4f} accuracy, "
            f"{summary['test']['macro_f1']:.4f} macro-F1."
        ),
        "base_weight": base_weight,
        "per_class_f1": {
            label: round(wild["per_class_f1"][label], 4) for label in LABELS
        },
        "_note": (
            f"Trained by train_audio.py on {summary['hyperparameters'].get('epochs')} "
            f"epochs max, best epoch {summary['best_epoch']}. The acted "
            f"domain scores markedly higher; see results/benchmark.json "
            f"for both."
        ),
    }

    dynamics = payload.setdefault("dynamics", {})
    dynamics.setdefault("audio_half_life_seconds", 8.0)
    dynamics.setdefault("audio_cutoff_seconds", 30.0)
    dynamics.setdefault("min_voiced_ratio", 0.10)

    RELIABILITY_PATH.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(f"  wrote audio profile to {RELIABILITY_PATH.name}")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weight", type=float, default=0.9)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    device = torch.device(
        arguments.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 62)
    print("AUDIO MODULE BENCHMARK")
    print("=" * 62)

    results = []

    for model_key in MODELS:
        path = checkpoint_path(model_key) / "summary.json"

        if not path.exists():
            print(f"  {model_key}: not trained, skipping")
            continue

        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["latency"] = measure_latency(model_key, device)

        results.append(summary)

    if not results:
        print("\nNothing trained yet. Run train_audio.py first.")
        return 1

    # ----------------------------------------------------
    # The comparison table
    # ----------------------------------------------------

    print(
        f"\n  {'model':<26} {'params':>12} {'acc':>7} {'macroF1':>8} "
        f"{'wtdF1':>7} {'ECE':>7} {'ms':>7}"
    )

    for summary in results:
        test = summary["test"]

        print(
            f"  {summary['display_name']:<26} "
            f"{summary['parameters']:>12,} "
            f"{test['accuracy']:>7.4f} {test['macro_f1']:>8.4f} "
            f"{test['weighted_f1']:>7.4f} {test['ece']:>7.4f} "
            f"{summary['latency']['median_ms']:>7.1f}"
        )

    # ----------------------------------------------------
    # By domain - the number that matters for deployment
    # ----------------------------------------------------

    print("\n  by domain")
    print(f"  {'model':<26} {'domain':<8} {'acc':>7} {'macroF1':>8} {'n':>6}")

    for summary in results:
        for domain, scores in sorted(summary["test_by_domain"].items()):
            print(
                f"  {summary['display_name']:<26} {domain:<8} "
                f"{scores['accuracy']:>7.4f} {scores['macro_f1']:>8.4f} "
                f"{scores['support']:>6d}"
            )

    # ----------------------------------------------------
    # Per-class F1 of the best model
    # ----------------------------------------------------

    best = max(results, key=lambda row: row["test"]["macro_f1"])

    print(f"\n  per-class F1, {best['display_name']}")
    print(f"  {'class':<10} {'overall':>8} {'acted':>8} {'wild':>8}")

    for label in LABELS:
        acted = best["test_by_domain"].get("acted", {})
        wild = best["test_by_domain"].get("wild", {})

        print(
            f"  {label:<10} "
            f"{best['test']['per_class_f1'][label]:>8.4f} "
            f"{acted.get('per_class_f1', {}).get(label, float('nan')):>8.4f} "
            f"{wild.get('per_class_f1', {}).get(label, float('nan')):>8.4f}"
        )

    payload = {
        "labels": list(LABELS),
        "device": str(device),
        "best_model": best["model"],
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\n  wrote {RESULTS_FILE}")

    if not arguments.no_publish:
        publish_reliability(best, arguments.base_weight)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
